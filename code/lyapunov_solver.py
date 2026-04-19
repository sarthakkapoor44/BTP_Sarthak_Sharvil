from solver_base import BaseSolver,SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from typing import Dict,Tuple
from objective_calculator import ObjectiveCalculator

class LyapunovSolver(BaseSolver):
    """
    Online Lyapunov Drift-Plus-Penalty solver.

    ✔ ADD + REMOVE actions
    ✔ Capacity-safe
    ✔ Adaptive V(t)
    ✔ MILP-consistent nominal objective computation
    ✔ Fully self-contained
    """

    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        super().__init__(config, data_gen)
        self.V0 = getattr(config, "lyapunov_V", 0.5)
        self.migration_budget = getattr(config, "lyapunov_budget", float("inf"))
        self.calculator = ObjectiveCalculator(config, data_gen)

    # ---------------------------------------------------------------------
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        sol = SlotSolution(time_slot=t, status="Feasible")

        # Adaptive Lyapunov weight
        V = self.V0 * (1.0 + 0.01 * t)

        # =============================================================
        # 1) Virtual Queues
        # =============================================================

        # ---- Capacity pressure
        Q_cap = {}
        for j in J:
            used = sum(
                self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                for i in I
            )
            Q_cap[j] = max(0.0, used - self.config.server_capacities[j])

        # ---- Coverage + Benefit pressure
        Q_cov, Q_B = {}, {}

        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue

            H_i = self.config.hop_budgets[i]

            for p in P_t:
                n_ip = self.data.counts.get((i, p, t), 0)
                if n_ip <= 0:
                    continue

                best_ring = None
                for d in range(0, H_i + 1):
                    if any(
                        A_prev.get((i, j), 0) == 1 and
                        self.config.hop_distances.get((j, p), float("inf")) <= d
                        for j in J
                    ):
                        best_ring = d
                        break

                if best_ring is None:
                    Q_cov[(i, p)] = 1.0
                    Q_B[(i, p)] = H_i
                else:
                    Q_cov[(i, p)] = 0.0
                    Q_B[(i, p)] = max(0.0, H_i - best_ring)

        # Start from previous placement and enforce activity semantics from D1-D2:
        # inactive datasets cannot remain placed in the new state.
        x_new = dict(A_prev)
        for i in I:
            if self.data.active_datasets.get((i, t), 0) == 1:
                continue
            for j in J:
                if x_new.get((i, j), 0) == 1:
                    x_new[(i, j)] = 0
                    sol.removes[(i, j)] = 1

        # =============================================================
        # 2) Score ADD and REMOVE actions
        # =============================================================
        actions = []

        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue

            size_i = self.config.dataset_sizes[i]
            H_i = self.config.hop_budgets[i]

            for j in J:
                # ---- Marginal benefit if (i,j) is active
                benefit_gain = 0.0
                for p in P_t:
                    if (i, p) not in Q_cov:
                        continue
                    if self.config.hop_distances.get((j, p), float("inf")) <= H_i:
                        benefit_gain += (
                            Q_cov[(i, p)] +
                            self.data.weights_nominal.get((i, p, t), 1.0) * Q_B[(i, p)]
                        )

                cap_penalty = Q_cap[j] * size_i
                add_cost = self.config.lambda_add * self.config.get_add_cost(i, j, t)

                # ---------- ADD ----------
                if A_prev.get((i, j), 0) == 0:
                    score = benefit_gain - cap_penalty - V * add_cost
                    actions.append(("add", score, i, j))

                # ---------- REMOVE ----------
                else:
                    # Objective calculator charges operation cost only for adds.
                    score = cap_penalty - benefit_gain
                    actions.append(("remove", score, i, j))

        # =============================================================
        # 3) Apply Actions (Greedy + Budget)
        # =============================================================
        actions.sort(key=lambda x: x[1], reverse=True)

        budget = self.migration_budget

        for act, score, i, j in actions:
            if score <= 0 or budget <= 0:
                break

            cost = self.config.get_add_cost(i, j, t)

            if act == "add" and x_new.get((i, j), 0) == 0:
                used = sum(
                    self.config.dataset_sizes[k] * x_new.get((k, j), 0)
                    for k in I
                )
                if used + self.config.dataset_sizes[i] > self.config.server_capacities[j]:
                    continue

                x_new[(i, j)] = 1
                sol.adds[(i, j)] = 1
                budget -= cost

            elif act == "remove" and x_new.get((i, j), 0) == 1:
                x_new[(i, j)] = 0
                sol.removes[(i, j)] = 1

        # Fill states
        for i in I:
            for j in J:
                if x_new.get((i, j), 0) == 1:
                    sol.states[(i, j)] = 1

        # =============================================================
        # 4) Compute MILP-Consistent Objective using centralized calculator
        # =============================================================
        
        breakdown = self.calculator.calculate(
            t=t,
            state_current=x_new,
            state_prev=A_prev
        )

        # ---- Final objective
        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.R_wc = float(breakdown.R_wc)
        sol.B_wc = float(breakdown.B_wc)
        sol.Op_cost = float(breakdown.Op_cost)
        sol.objective_value = (breakdown.objective_total if self.config.use_robust 
                              else breakdown.objective_nominal)

        return sol
