from solver_base import BaseSolver, SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator
from typing import Dict, Tuple, List
import time
from dataclasses import dataclass, field


@dataclass
class BeamState:
    placement: Dict[Tuple[int, int], int]
    adds: Dict[Tuple[int, int], int] = field(default_factory=dict)
    removes: Dict[Tuple[int, int], int] = field(default_factory=dict)
    touched_pairs: set = field(default_factory=set)
    score: float = float("-inf")
    breakdown: object = None


class BeamSearchSolver(BaseSolver):
    """
    Online constraint-aware beam search solver.

    This solver is intentionally different from MILP/RL:
    - It searches short sequences of add/remove moves within each slot.
    - Every candidate is evaluated with the exact centralized objective.
    - Hard constraints are enforced through exact feasibility checks.
    - A final repair pass guarantees coverage before the slot is accepted.

    It is designed as a practical middle ground between greedy heuristics and exact MILP.
    """

    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        super().__init__(config, data_gen)
        self.calculator = ObjectiveCalculator(config, data_gen)
        self.beam_width = int(getattr(config, "beam_width", 5))
        self.max_depth = int(getattr(config, "beam_max_depth", 4))
        self.max_candidates_per_step = int(getattr(config, "beam_candidates_per_step", 8))
        self.allow_remove = bool(getattr(config, "beam_allow_remove", True))

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        start_time = time.time()
        sol = SlotSolution(time_slot=t, status="Feasible")
        initial_state = dict(A_prev)

        if bool(getattr(self.config, "enforce_coverage_repair", True)):
            initial_state = self._repair_coverage(t, initial_state, A_prev, sol)

        current_breakdown = self.calculator.calculate(t=t, state_current=initial_state, state_prev=A_prev)
        best_state = dict(initial_state)
        best_breakdown = current_breakdown
        best_score = self._score_breakdown(current_breakdown)

        beam: List[BeamState] = [BeamState(placement=dict(initial_state), score=best_score, breakdown=current_breakdown)]
        visited = {self._state_signature(initial_state)}

        for _depth in range(self.max_depth):
            next_beam: List[BeamState] = []
            for state in beam:
                for action in self._candidate_actions(t, state.placement, A_prev, state.touched_pairs):
                    next_placement, action_type, pair = action
                    sig = self._state_signature(next_placement)
                    if sig in visited:
                        continue
                    visited.add(sig)

                    if bool(getattr(self.config, "enforce_coverage_repair", True)) and not self._nominal_feasible_after_change(t, next_placement):
                        continue

                    breakdown = self.calculator.calculate(t=t, state_current=next_placement, state_prev=A_prev)
                    score = self._score_breakdown(breakdown)
                    new_state = BeamState(
                        placement=next_placement,
                        adds=dict(state.adds),
                        removes=dict(state.removes),
                        touched_pairs=set(state.touched_pairs),
                        score=score,
                        breakdown=breakdown,
                    )
                    if action_type == "add":
                        new_state.adds[pair] = 1
                    else:
                        new_state.removes[pair] = 1
                    new_state.touched_pairs.add(pair)
                    next_beam.append(new_state)

                    if score > best_score:
                        best_score = score
                        best_state = dict(next_placement)
                        best_breakdown = breakdown

            if not next_beam:
                break

            next_beam.sort(key=lambda s: s.score, reverse=True)
            beam = next_beam[: self.beam_width]

        final_state = dict(best_state)
        if bool(getattr(self.config, "enforce_coverage_repair", True)):
            final_state = self._repair_coverage(t, final_state, A_prev, sol)

        final_breakdown = self.calculator.calculate(t=t, state_current=final_state, state_prev=A_prev)
        sol.states = {k: v for k, v in final_state.items() if v == 1}
        sol.adds = dict(best_adds := self._diff_adds(A_prev, final_state))
        sol.removes = dict(best_removes := self._diff_removes(A_prev, final_state))
        sol.R_nominal = float(final_breakdown.R_nominal)
        sol.B_nominal = float(final_breakdown.B_nominal)
        sol.R_wc = float(final_breakdown.R_wc)
        sol.B_wc = float(final_breakdown.B_wc)
        sol.Op_cost = float(final_breakdown.Op_cost)
        sol.objective_value = float(final_breakdown.objective_total if self.config.use_robust else final_breakdown.objective_nominal)
        sol.solve_time = time.time() - start_time
        sol.status = "Feasible"
        return sol

    def _score_breakdown(self, breakdown) -> float:
        return float(breakdown.objective_total if self.config.use_robust else breakdown.objective_nominal)

    def _state_signature(self, placement: Dict[Tuple[int, int], int]) -> Tuple[int, ...]:
        return tuple(int(placement.get((i, j), 0)) for i in range(self.config.num_datasets) for j in range(self.config.num_servers))

    def _diff_adds(self, before: Dict[Tuple[int, int], int], after: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        adds = {}
        for i in range(self.config.num_datasets):
            for j in range(self.config.num_servers):
                if before.get((i, j), 0) == 0 and after.get((i, j), 0) == 1:
                    adds[(i, j)] = 1
        return adds

    def _diff_removes(self, before: Dict[Tuple[int, int], int], after: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
        removes = {}
        for i in range(self.config.num_datasets):
            for j in range(self.config.num_servers):
                if before.get((i, j), 0) == 1 and after.get((i, j), 0) == 0:
                    removes[(i, j)] = 1
        return removes

    def _candidate_actions(self, t: int, placement: Dict[Tuple[int, int], int], prev_state: Dict[Tuple[int, int], int], touched_pairs: set):
        scored_actions = []
        for i in range(self.config.num_datasets):
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            for j in range(self.config.num_servers):
                if (i, j) in touched_pairs:
                    continue
                if placement.get((i, j), 0) == 0 and self._add_feasible(t, placement, i, j):
                    candidate = dict(placement)
                    candidate[(i, j)] = 1
                    breakdown = self.calculator.calculate(t=t, state_current=candidate, state_prev=prev_state)
                    scored_actions.append((self._score_breakdown(breakdown), "add", (i, j), candidate))
                if self.allow_remove and placement.get((i, j), 0) == 1 and self._remove_feasible(t, placement, i, j):
                    candidate = dict(placement)
                    candidate[(i, j)] = 0
                    breakdown = self.calculator.calculate(t=t, state_current=candidate, state_prev=prev_state)
                    scored_actions.append((self._score_breakdown(breakdown), "remove", (i, j), candidate))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        for _, action_type, pair, candidate in scored_actions[: self.max_candidates_per_step]:
            yield candidate, action_type, pair

    def _repair_coverage(self, t: int, placement: Dict[Tuple[int, int], int], prev_state: Dict[Tuple[int, int], int], sol: SlotSolution):
        current = dict(placement)
        max_steps = self.config.num_datasets * self.config.num_servers
        for _ in range(max_steps):
            uncovered = self._uncovered_demand_pairs(t, current)
            if uncovered <= 0:
                break

            best_candidate = None
            best_score = float("-inf")
            for i in range(self.config.num_datasets):
                for j in range(self.config.num_servers):
                    if current.get((i, j), 0) == 1:
                        continue
                    if not self._add_feasible(t, current, i, j):
                        continue
                    candidate = dict(current)
                    candidate[(i, j)] = 1
                    if self._uncovered_demand_pairs(t, candidate) >= uncovered:
                        continue
                    breakdown = self.calculator.calculate(t=t, state_current=candidate, state_prev=prev_state)
                    score = self._score_breakdown(breakdown)
                    if score > best_score:
                        best_score = score
                        best_candidate = ((i, j), candidate)

            if best_candidate is None:
                break

            (i, j), candidate = best_candidate
            current = candidate
            sol.adds[(i, j)] = 1

        if self._uncovered_demand_pairs(t, current) > 0:
            # Final hard repair: if still uncovered, keep adding any feasible covering replica.
            for _ in range(max_steps):
                uncovered = self._uncovered_demand_pairs(t, current)
                if uncovered <= 0:
                    break
                best_pair = None
                best_drop = 0
                for i in range(self.config.num_datasets):
                    for j in range(self.config.num_servers):
                        if current.get((i, j), 0) == 1:
                            continue
                        if not self._add_feasible(t, current, i, j):
                            continue
                        candidate = dict(current)
                        candidate[(i, j)] = 1
                        drop = uncovered - self._uncovered_demand_pairs(t, candidate)
                        if drop > best_drop:
                            best_drop = drop
                            best_pair = (i, j)
                if best_pair is None:
                    break
                current[best_pair] = 1
                sol.adds[best_pair] = 1

        return current

    def _coverage_count(self, i: int, p: int, placement: Dict[Tuple[int, int], int]) -> int:
        H_i = self.config.hop_budgets[i]
        count = 0
        for j in range(self.config.num_servers):
            if placement.get((i, j), 0) != 1:
                continue
            if self.config.hop_distances.get((j, p), float("inf")) <= H_i:
                count += 1
        return count

    def _uncovered_demand_pairs(self, t: int, placement: Dict[Tuple[int, int], int]) -> int:
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

    def _nominal_feasible_after_change(self, t: int, placement: Dict[Tuple[int, int], int]) -> bool:
        for i in range(self.config.num_datasets):
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            for p in self.data.attachment_points.get(t, []):
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue
                if self._coverage_count(i, p, placement) <= 0:
                    return False
        return True

    def _add_feasible(self, t: int, current_state: Dict[Tuple[int, int], int], i: int, j: int) -> bool:
        if self.data.active_datasets.get((i, t), 0) != 1:
            return False
        if current_state.get((i, j), 0) == 1:
            return False
        H_i = self.config.hop_budgets[i]
        if not any(
            self.data.counts.get((i, p, t), 0) > 0 and self.config.hop_distances.get((j, p), float("inf")) <= H_i
            for p in self.data.attachment_points.get(t, [])
        ):
            return False
        size_i = self.config.dataset_sizes[i]
        current_load = sum(self.config.dataset_sizes[ii] * current_state.get((ii, j), 0) for ii in range(self.config.num_datasets))
        if current_load + size_i > self.config.server_capacities[j]:
            return False
        return True

    def _remove_feasible(self, t: int, current_state: Dict[Tuple[int, int], int], i: int, j: int) -> bool:
        if current_state.get((i, j), 0) != 1:
            return False
        candidate = dict(current_state)
        candidate[(i, j)] = 0
        return self._nominal_feasible_after_change(t, candidate)
