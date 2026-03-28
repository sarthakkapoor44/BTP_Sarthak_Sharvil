"""Auto-extracted from temp.ipynb cell 40."""

# =============================================================================
# Q-LEARNING SOLVER (ONLINE, NO LOOK-AHEAD BIAS)
# =============================================================================
import random
from collections import defaultdict
from typing import Dict, Tuple
from solver_base import BaseSolver, SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator


class QLearningSolver(BaseSolver):
    """
    Tabular Q-learning solver with NO look-ahead bias.

    Actions:
        0 = do nothing
        1 = add one replica (chosen heuristically)
        2 = remove one replica (chosen heuristically)

    Reward:
        Immediate nominal objective at slot t
    """

    def __init__(
        self,
        config: uEDDEConfig,
        data_gen: DataGenerator,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        super().__init__(config, data_gen)
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration
        self.Q = defaultdict(float)
        self.calculator = ObjectiveCalculator(config, data_gen)

    # ------------------------------------------------------------------
    # State encoding (CURRENT SLOT ONLY)
    # ------------------------------------------------------------------
    def _encode_state(self, Q_cov, Q_B, Q_cap, A_prev):
        """
        Compact, discrete state representation.
        No future info.
        """
        def bucket(x, step):
            return int(x // step)

        return (
            bucket(sum(Q_cov.values()), 1),
            bucket(sum(Q_B.values()), 2),
            bucket(sum(Q_cap.values()), 5),
            bucket(sum(A_prev.values()), 3),
        )

    # ------------------------------------------------------------------
    # ε-greedy action selection
    # ------------------------------------------------------------------
    def _select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        return max([0, 1, 2], key=lambda a: self.Q[(state, a)])

    # ------------------------------------------------------------------
    # Heuristic add/remove selectors (CURRENT SLOT ONLY)
    # ------------------------------------------------------------------
    def _choose_add(self, t, A_prev):
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        best_gain = -1
        best_pair = None

        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            H_i = self.config.hop_budgets[i]

            for j in J:
                if A_prev.get((i, j), 0) == 1:
                    continue

                gain = 0
                for p in P_t:
                    if self.data.counts.get((i, p, t), 0) <= 0:
                        continue
                    if self.config.hop_distances.get((j, p), float("inf")) <= H_i:
                        gain += self.data.counts[(i, p, t)]

                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)

        return best_pair

    def _choose_remove(self, t, A_prev):
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)

        worst_cost = float("inf")
        worst_pair = None

        for i in I:
            for j in J:
                if A_prev.get((i, j), 0) == 1:
                    cost = self.config.get_add_cost(i, j, t)
                    if cost < worst_cost:
                        worst_cost = cost
                        worst_pair = (i, j)

        return worst_pair

    # ------------------------------------------------------------------
    # Main solve_slot (ONLINE, unbiased)
    # ------------------------------------------------------------------
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        sol = SlotSolution(time_slot=t, status="Feasible")

        # --------------------------------------------------------------
        # 1) Compute CURRENT virtual signals (no future info)
        # --------------------------------------------------------------
        Q_cap, Q_cov, Q_B = {}, {}, {}

        for j in J:
            used = sum(
                self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                for i in I
            )
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
                        A_prev.get((i, j), 0) == 1 and
                        self.config.hop_distances.get((j, p), float("inf")) <= d
                        for j in J
                    ):
                        best_ring = d
                        break

                Q_cov[(i, p)] = 0 if best_ring is not None else 1
                Q_B[(i, p)] = 0 if best_ring is None else max(0, H_i - best_ring)

        # --------------------------------------------------------------
        # 2) RL action (NO look-ahead)
        # --------------------------------------------------------------
        state = self._encode_state(Q_cov, Q_B, Q_cap, A_prev)
        action = self._select_action(state)

        x_new = dict(A_prev)

        if action == 1:
            pair = self._choose_add(t, A_prev)
            if pair:
                x_new[pair] = 1
                sol.adds[pair] = 1

        elif action == 2:
            pair = self._choose_remove(t, A_prev)
            if pair:
                x_new[pair] = 0
                sol.removes[pair] = 1

        for i in I:
            for j in J:
                if x_new.get((i, j), 0) == 1:
                    sol.states[(i, j)] = 1

        # --------------------------------------------------------------
        # 3) Immediate reward = REALIZED slot-t objective (via calculator)
        # --------------------------------------------------------------
        breakdown = self.calculator.calculate(
            t=t,
            state_current=x_new,
            state_prev=A_prev
        )
        
        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.R_wc = float(breakdown.R_wc)
        sol.B_wc = float(breakdown.B_wc)
        sol.Op_cost = float(breakdown.Op_cost)
        
        reward = (breakdown.objective_total if self.config.use_robust 
                 else breakdown.objective_nominal)
        sol.objective_value = reward

        # --------------------------------------------------------------
        # 4) Q update (standard, unbiased)
        # --------------------------------------------------------------
        next_state = self._encode_state(Q_cov, Q_B, Q_cap, x_new)
        best_next = max(self.Q[(next_state, a)] for a in [0, 1, 2])

        self.Q[(state, action)] += self.alpha * (
            reward + self.gamma * best_next - self.Q[(state, action)]
        )

        return sol

    # NOTE: _compute_nominal_objective has been removed in favor of using ObjectiveCalculator
