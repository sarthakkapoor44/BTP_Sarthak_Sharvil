"""Auto-extracted from temp.ipynb cell 40."""

# =============================================================================
# Q-LEARNING SOLVER (ONLINE, NO LOOK-AHEAD BIAS)
# =============================================================================
import random
import time
from collections import defaultdict
from typing import Dict, Tuple

from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator
from solver_base import BaseSolver, SlotSolution


class QLearningSolver(BaseSolver):
    """
    Constrained sequential Q-learning solver with no look-ahead bias.

    Actions:
        0 = stop
        1 = add one replica
        2 = remove one replica

    The solver now performs multiple atomic edits per slot until no feasible
    positive-gain action remains. Each candidate action is masked by current
    feasibility, adds are limited by capacity and ingress, and idle replicas are
    softly penalized so they can be pruned when they no longer serve demand.
    """

    def __init__(
        self,
        config: uEDDEConfig,
        data_gen: DataGenerator,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        super().__init__(config, data_gen)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)
        self.calculator = ObjectiveCalculator(config, data_gen)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _build_virtual_signals(self, t: int, placement: Dict[Tuple[int, int], int]):
        """Build coarse state signals from the current placement."""
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        Q_cap, Q_cov, Q_B = {}, {}, {}

        for j in J:
            used = sum(self.config.dataset_sizes[i] * placement.get((i, j), 0) for i in I)
            Q_cap[j] = max(0.0, used - self.config.server_capacities[j])

        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue

            H_i = self.config.hop_budgets[i]
            for p in P_t:
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue

                best_ring = None
                for d in range(H_i + 1):
                    if any(
                        placement.get((i, j), 0) == 1
                        and self.config.hop_distances.get((j, p), float("inf")) <= d
                        for j in J
                    ):
                        best_ring = d
                        break

                Q_cov[(i, p)] = 0 if best_ring is not None else 1
                Q_B[(i, p)] = 0 if best_ring is None else max(0, H_i - best_ring)

        return Q_cov, Q_B, Q_cap

    def _encode_state(self, Q_cov, Q_B, Q_cap, placement):
        """Compact, discrete state representation using current-slot information."""

        def bucket(x, step):
            return int(x // step)

        return (
            bucket(sum(Q_cov.values()), 1),
            bucket(sum(Q_B.values()), 2),
            bucket(sum(Q_cap.values()), 5),
            bucket(sum(placement.values()), 3),
        )

    def _compute_slot_loads(self, placement: Dict[Tuple[int, int], int]):
        return {
            j: sum(
                self.config.dataset_sizes[i] * placement.get((i, j), 0)
                for i in range(self.config.num_datasets)
            )
            for j in range(self.config.num_servers)
        }

    def _replica_serves_any_demand(
        self,
        i: int,
        j: int,
        t: int,
        placement: Dict[Tuple[int, int], int],
    ) -> bool:
        """Return True if replica (i, j) covers at least one demanded AP in slot t."""
        if self.data.active_datasets.get((i, t), 0) != 1:
            return False

        H_i = self.config.hop_budgets[i]
        for p in self.data.attachment_points.get(t, []):
            if self.data.counts.get((i, p, t), 0) <= 0:
                continue
            if self.config.hop_distances.get((j, p), float("inf")) <= H_i:
                return True
        return False

    def _idle_replica_penalty(self, t: int, placement: Dict[Tuple[int, int], int]) -> int:
        """Count replicas that do not serve any current-slot demand."""
        penalty = 0
        for (i, j), value in placement.items():
            if value != 1:
                continue
            if not self._replica_serves_any_demand(i, j, t, placement):
                penalty += 1
        return penalty

    def _uncovered_demand_pairs(self, t: int, placement: Dict[Tuple[int, int], int]) -> int:
        """Count demanded (dataset, AP) pairs that have no covering replica within hop budget."""
        uncovered = 0
        for i in range(self.config.num_datasets):
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            for p in self.data.attachment_points.get(t, []):
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue
                if self._coverage_count(i, p, placement) <= 0:
                    uncovered += 1
        return uncovered

    def _shaped_objective(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        prev_state: Dict[Tuple[int, int], int],
    ):
        """MILP-aligned objective with explicit penalties for poor service state."""
        breakdown = self.calculator.calculate(
            t=t,
            state_current=current_state,
            state_prev=prev_state,
        )
        base_value = breakdown.objective_total if self.config.use_robust else breakdown.objective_nominal
        idle_weight = getattr(self.config, "eta_stability", 0.0)
        if idle_weight <= 0:
            idle_weight = 1e-3
        idle_penalty = idle_weight * self._idle_replica_penalty(t, current_state)
        uncovered_weight = float(getattr(self.config, "uncovered_pair_penalty", 1.0))
        uncovered_penalty = uncovered_weight * self._uncovered_demand_pairs(t, current_state)
        return breakdown, base_value - idle_penalty - uncovered_penalty

    def _force_coverage_repair(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        prev_state: Dict[Tuple[int, int], int],
        touched_pairs: set,
        slot_add_usage: Dict[int, float],
        sol: SlotSolution,
    ):
        """Greedily add replicas until all demanded pairs are covered or no feasible add exists."""
        max_repair_steps = self.config.num_datasets * self.config.num_servers

        for _ in range(max_repair_steps):
            uncovered_before = self._uncovered_demand_pairs(t, current_state)
            if uncovered_before <= 0:
                break

            best_pair = None
            best_uncovered_drop = 0
            best_delta = float("-inf")
            _, current_value = self._shaped_objective(t, current_state, prev_state)

            for i in range(self.config.num_datasets):
                for j in range(self.config.num_servers):
                    if current_state.get((i, j), 0) == 1:
                        continue
                    if not self._add_feasible_basic(t, current_state, slot_add_usage, i, j):
                        continue

                    candidate = dict(current_state)
                    candidate[(i, j)] = 1
                    uncovered_after = self._uncovered_demand_pairs(t, candidate)
                    uncovered_drop = uncovered_before - uncovered_after
                    if uncovered_drop <= 0:
                        continue

                    _, candidate_value = self._shaped_objective(t, candidate, prev_state)
                    delta = candidate_value - current_value

                    if (uncovered_drop > best_uncovered_drop) or (
                        uncovered_drop == best_uncovered_drop and delta > best_delta
                    ):
                        best_pair = (i, j)
                        best_uncovered_drop = uncovered_drop
                        best_delta = delta

            if best_pair is None:
                break

            i, j = best_pair
            current_state[best_pair] = 1
            sol.adds[best_pair] = 1
            touched_pairs.add(best_pair)
            slot_add_usage[j] += self.config.dataset_sizes[i]

        # If some uncovered pairs still remain, do one final pass that ignores shaped reward
        # and only tries to eliminate uncovered demand with any feasible add.
        uncovered_remaining = self._uncovered_demand_pairs(t, current_state)
        if uncovered_remaining > 0:
            for _ in range(max_repair_steps):
                uncovered_before = self._uncovered_demand_pairs(t, current_state)
                if uncovered_before <= 0:
                    break

                best_pair = None
                best_uncovered_drop = 0

                for i in range(self.config.num_datasets):
                    for j in range(self.config.num_servers):
                        if current_state.get((i, j), 0) == 1:
                            continue
                        if not self._add_feasible_basic(t, current_state, slot_add_usage, i, j):
                            continue

                        candidate = dict(current_state)
                        candidate[(i, j)] = 1
                        uncovered_after = self._uncovered_demand_pairs(t, candidate)
                        uncovered_drop = uncovered_before - uncovered_after
                        if uncovered_drop > best_uncovered_drop:
                            best_pair = (i, j)
                            best_uncovered_drop = uncovered_drop

                if best_pair is None or best_uncovered_drop <= 0:
                    break

                i, j = best_pair
                current_state[best_pair] = 1
                sol.adds[best_pair] = 1
                touched_pairs.add(best_pair)
                slot_add_usage[j] += self.config.dataset_sizes[i]

    def _coverage_count(self, i: int, p: int, placement: Dict[Tuple[int, int], int]) -> int:
        H_i = self.config.hop_budgets[i]
        count = 0
        for j in range(self.config.num_servers):
            if placement.get((i, j), 0) != 1:
                continue
            if self.config.hop_distances.get((j, p), float("inf")) <= H_i:
                count += 1
        return count

    def _nominal_feasible_after_change(
        self,
        t: int,
        placement: Dict[Tuple[int, int], int],
    ) -> bool:
        """Mask actions that would break nominal coverage for the current slot."""
        for i in range(self.config.num_datasets):
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            for p in self.data.attachment_points.get(t, []):
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue
                if self._coverage_count(i, p, placement) <= 0:
                    return False
        return True

    def _add_feasible_basic(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        slot_add_usage: Dict[int, float],
        i: int,
        j: int,
    ) -> bool:
        if self.data.active_datasets.get((i, t), 0) != 1:
            return False
        if current_state.get((i, j), 0) == 1:
            return False

        if not self._replica_serves_any_demand(i, j, t, current_state):
            return False

        size_i = self.config.dataset_sizes[i]
        current_load = self._compute_slot_loads(current_state)[j]
        if current_load + size_i > self.config.server_capacities[j]:
            return False

        if self.config.use_ingress_constraint:
            if slot_add_usage.get(j, 0.0) + size_i > self.config.server_bandwidth[j]:
                return False

        return True

    def _add_feasible(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        slot_add_usage: Dict[int, float],
        i: int,
        j: int,
    ) -> bool:
        """Feasible add for search/repair: supports stepwise restoration from uncovered states."""
        return self._add_feasible_basic(t, current_state, slot_add_usage, i, j)

    def _remove_feasible(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        i: int,
        j: int,
    ) -> bool:
        if current_state.get((i, j), 0) != 1:
            return False

        candidate = dict(current_state)
        candidate[(i, j)] = 0
        return self._nominal_feasible_after_change(t, candidate)

    def _best_add_candidate(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        prev_state: Dict[Tuple[int, int], int],
        touched_pairs: set,
        slot_add_usage: Dict[int, float],
    ):
        current_breakdown, current_value = self._shaped_objective(t, current_state, prev_state)
        best_pair = None
        best_delta = 0.0

        for i in range(self.config.num_datasets):
            for j in range(self.config.num_servers):
                if (i, j) in touched_pairs:
                    continue
                if not self._add_feasible(t, current_state, slot_add_usage, i, j):
                    continue

                candidate = dict(current_state)
                candidate[(i, j)] = 1
                _, candidate_value = self._shaped_objective(t, candidate, prev_state)
                delta = candidate_value - current_value
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j)

        return best_pair, best_delta, current_breakdown, current_value

    def _best_remove_candidate(
        self,
        t: int,
        current_state: Dict[Tuple[int, int], int],
        prev_state: Dict[Tuple[int, int], int],
        touched_pairs: set,
    ):
        current_breakdown, current_value = self._shaped_objective(t, current_state, prev_state)
        best_pair = None
        best_delta = 0.0

        for i in range(self.config.num_datasets):
            for j in range(self.config.num_servers):
                if (i, j) in touched_pairs:
                    continue
                if not self._remove_feasible(t, current_state, i, j):
                    continue

                candidate = dict(current_state)
                candidate[(i, j)] = 0
                _, candidate_value = self._shaped_objective(t, candidate, prev_state)
                delta = candidate_value - current_value
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j)

        return best_pair, best_delta, current_breakdown, current_value

    def _select_action_type(self, state, action_scores):
        available_actions = [a for a in [0, 1, 2] if action_scores.get(a) is not None]
        if not available_actions:
            return 0

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        return max(available_actions, key=lambda a: self.Q[(state, a)] + action_scores[a])

    # ------------------------------------------------------------------
    # Main solve_slot (ONLINE, unbiased)
    # ------------------------------------------------------------------
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        sol = SlotSolution(time_slot=t, status="Feasible")
        current_state = dict(A_prev)
        touched_pairs = set()
        slot_add_usage = {j: 0.0 for j in range(self.config.num_servers)}
        max_steps = max(1, self.config.num_datasets * self.config.num_servers * 2)

        if bool(getattr(self.config, "enforce_coverage_repair", True)):
            self._force_coverage_repair(
                t=t,
                current_state=current_state,
                prev_state=A_prev,
                touched_pairs=touched_pairs,
                slot_add_usage=slot_add_usage,
                sol=sol,
            )

        current_breakdown, current_value = self._shaped_objective(t, current_state, A_prev)

        for _ in range(max_steps):
            Q_cov, Q_B, Q_cap = self._build_virtual_signals(t, current_state)
            state = self._encode_state(Q_cov, Q_B, Q_cap, current_state)

            best_add_pair, best_add_delta, _, _ = self._best_add_candidate(
                t, current_state, A_prev, touched_pairs, slot_add_usage
            )
            best_remove_pair, best_remove_delta, _, _ = self._best_remove_candidate(
                t, current_state, A_prev, touched_pairs
            )

            action_scores = {
                0: 0.0,
                1: best_add_delta if best_add_pair is not None else None,
                2: best_remove_delta if best_remove_pair is not None else None,
            }

            action_type = self._select_action_type(state, action_scores)
            if action_type == 0:
                break

            if action_type == 1:
                pair = best_add_pair
                if pair is None or best_add_delta <= 1e-9:
                    break

                i, j = pair
                next_state = dict(current_state)
                next_state[pair] = 1
                next_breakdown, next_value = self._shaped_objective(t, next_state, A_prev)
                reward = next_value - current_value

                next_Q_cov, next_Q_B, next_Q_cap = self._build_virtual_signals(t, next_state)
                next_state_key = self._encode_state(next_Q_cov, next_Q_B, next_Q_cap, next_state)
                best_next = max(self.Q[(next_state_key, a)] for a in [0, 1, 2])
                self.Q[(state, action_type)] += self.alpha * (
                    reward + self.gamma * best_next - self.Q[(state, action_type)]
                )

                current_state = next_state
                current_value = next_value
                current_breakdown = next_breakdown
                sol.adds[pair] = 1
                touched_pairs.add(pair)
                slot_add_usage[j] += self.config.dataset_sizes[i]
                continue

            if action_type == 2:
                pair = best_remove_pair
                if pair is None or best_remove_delta <= 1e-9:
                    break

                next_state = dict(current_state)
                next_state[pair] = 0
                next_breakdown, next_value = self._shaped_objective(t, next_state, A_prev)
                reward = next_value - current_value

                next_Q_cov, next_Q_B, next_Q_cap = self._build_virtual_signals(t, next_state)
                next_state_key = self._encode_state(next_Q_cov, next_Q_B, next_Q_cap, next_state)
                best_next = max(self.Q[(next_state_key, a)] for a in [0, 1, 2])
                self.Q[(state, action_type)] += self.alpha * (
                    reward + self.gamma * best_next - self.Q[(state, action_type)]
                )

                current_state = next_state
                current_value = next_value
                current_breakdown = next_breakdown
                sol.removes[pair] = 1
                touched_pairs.add(pair)
                continue

        if bool(getattr(self.config, "enforce_coverage_repair", True)):
            self._force_coverage_repair(
                t=t,
                current_state=current_state,
                prev_state=A_prev,
                touched_pairs=touched_pairs,
                slot_add_usage=slot_add_usage,
                sol=sol,
            )

        uncovered_after = self._uncovered_demand_pairs(t, current_state)
        if uncovered_after > 0:
            sol.status = f"InfeasibleCoverage(uncovered_pairs={uncovered_after})"

        for (i, j), value in current_state.items():
            if value == 1:
                sol.states[(i, j)] = 1

        final_breakdown = self.calculator.calculate(
            t=t,
            state_current=current_state,
            state_prev=A_prev,
        )

        sol.R_nominal = float(final_breakdown.R_nominal)
        sol.B_nominal = float(final_breakdown.B_nominal)
        sol.R_wc = float(final_breakdown.R_wc)
        sol.B_wc = float(final_breakdown.B_wc)
        sol.Op_cost = float(final_breakdown.Op_cost)
        sol.objective_value = (
            final_breakdown.objective_total if self.config.use_robust
            else final_breakdown.objective_nominal
        )

        return sol
