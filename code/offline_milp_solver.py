from solver_base import BaseSolver, SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator
from typing import Dict, Tuple
import pulp
import csv


class OfflineMILPSolver(BaseSolver):
    """
    Full-horizon (hindsight) MILP benchmark solver.

    This solver optimizes across all time slots jointly, using future demand known
    over the horizon. It is intended as an oracle-style benchmark against online
    controllers.

        Note:
    - The exact R_nom ratio term from per-slot online objective is fractional with
      variable denominator across time in the horizon model. To keep a tractable
      MILP, the optimization objective uses nominal benefit minus add-cost, with
      strict coverage constraints.
        - When use_robust=True, this solver adds a conservative robust surrogate:
            K-survivable coverage and a robust benefit term based on the (K+1)-th
            availability level inside hop rings.
    - Final reported per-slot metrics and objective are still computed with the
      centralized ObjectiveCalculator for fair reporting.
    """

    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        super().__init__(config, data_gen)
        self.model = None
        self.calculator = ObjectiveCalculator(config, data_gen)

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        raise NotImplementedError("OfflineMILPSolver optimizes the full horizon via solve_all().")

    def solve_all(self):
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        T = range(1, self.config.T + 1)

        self.model = pulp.LpProblem("uEDDE_offline_hindsight", pulp.LpMaximize)

        # Decision variables for every slot
        x = pulp.LpVariable.dicts("x", [(i, j, t) for i in I for j in J for t in T], cat="Binary")
        a = pulp.LpVariable.dicts("a", [(i, j, t) for i in I for j in J for t in T], cat="Binary")
        r = pulp.LpVariable.dicts("r", [(i, j, t) for i in I for j in J for t in T], cat="Binary")

        # Benefit ring variables
        w_nom = {}
        B_nom_ip = {}
        w_wc = {}
        B_wc_ip = {}
        for t in T:
            P_t = self.data.attachment_points.get(t, [])
            for i in I:
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    for d in range(0, H_i + 1):
                        w_nom[(i, p, d, t)] = pulp.LpVariable(f"w_nom_{i}_{p}_{d}_{t}", cat="Binary")
                    B_nom_ip[(i, p, t)] = pulp.LpVariable(
                        f"B_nom_{i}_{p}_{t}",
                        lowBound=0,
                        upBound=self.config.hop_budgets[i],
                    )
                    if self.config.use_robust:
                        for d in range(0, H_i + 1):
                            w_wc[(i, p, d, t)] = pulp.LpVariable(f"w_wc_{i}_{p}_{d}_{t}", cat="Binary")
                        B_wc_ip[(i, p, t)] = pulp.LpVariable(
                            f"B_wc_{i}_{p}_{t}",
                            lowBound=0,
                            upBound=self.config.hop_budgets[i],
                        )

        # Transition, feasibility, and capacity constraints across time
        for t in T:
            for i in I:
                alpha = self.data.active_datasets.get((i, t), 0)
                for j in J:
                    prev = self.data.initial_state.get((i, j), 0) if t == 1 else x[(i, j, t - 1)]

                    # State transition
                    self.model += x[(i, j, t)] == alpha * (prev - r[(i, j, t)]) + a[(i, j, t)]

                    # Add/remove feasibility
                    self.model += r[(i, j, t)] <= prev
                    self.model += a[(i, j, t)] <= alpha
                    self.model += a[(i, j, t)] <= 1 - prev
                    self.model += a[(i, j, t)] + r[(i, j, t)] <= 1

            # Capacity and optional ingress
            for j in J:
                self.model += (
                    pulp.lpSum(self.config.dataset_sizes[i] * x[(i, j, t)] for i in I)
                    <= self.config.server_capacities[j]
                )
                if self.config.use_ingress_constraint:
                    self.model += (
                        pulp.lpSum(self.config.dataset_sizes[i] * a[(i, j, t)] for i in I)
                        <= self.config.server_bandwidth[j]
                    )

            # Coverage and ring constraints
            P_t = self.data.attachment_points.get(t, [])
            for i in I:
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    count = self.data.counts.get((i, p, t), 0)
                    L_bar = 1 if count >= 1 else 0
                    neighborhood = self.data.get_neighborhood(p, H_i)

                    # Robust multiplier m = K+1 bounded by neighborhood cardinality.
                    robust_mult = min(self.config.K_failures + 1, len(neighborhood)) if neighborhood else 0
                    required_copies = robust_mult if self.config.use_robust else 1

                    if L_bar > 0:
                        self.model += pulp.lpSum(x[(i, j, t)] for j in neighborhood) >= required_copies * L_bar

                    for d in range(0, H_i + 1):
                        ring_d = [
                            j for j in J
                            if self.config.hop_distances.get((j, p), float("inf")) <= d
                        ]
                        self.model += w_nom[(i, p, d, t)] <= pulp.lpSum(x[(i, j, t)] for j in ring_d)
                        self.model += w_nom[(i, p, d, t)] <= L_bar
                        if d > 0:
                            self.model += w_nom[(i, p, d - 1, t)] <= w_nom[(i, p, d, t)]

                        if self.config.use_robust:
                            # w_wc=1 implies at least robust_mult replicas available within ring d.
                            if robust_mult > 0:
                                self.model += w_wc[(i, p, d, t)] <= (1.0 / robust_mult) * pulp.lpSum(x[(i, j, t)] for j in ring_d)
                            else:
                                self.model += w_wc[(i, p, d, t)] <= 0
                            self.model += w_wc[(i, p, d, t)] <= L_bar
                            if d > 0:
                                self.model += w_wc[(i, p, d - 1, t)] <= w_wc[(i, p, d, t)]

                    self.model += (
                        B_nom_ip[(i, p, t)]
                        <= pulp.lpSum(w_nom[(i, p, d, t)] for d in range(0, H_i + 1)) - L_bar
                    )
                    self.model += B_nom_ip[(i, p, t)] >= 0

                    if self.config.use_robust:
                        self.model += (
                            B_wc_ip[(i, p, t)]
                            <= pulp.lpSum(w_wc[(i, p, d, t)] for d in range(0, H_i + 1)) - L_bar
                        )
                        self.model += B_wc_ip[(i, p, t)] >= 0

        # Full-horizon objective (linear MILP form): sum_t B_nom_t - Op_t
        obj_terms = []
        for t in T:
            P_t = self.data.attachment_points.get(t, [])
            I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
            H_sum = sum(self.config.hop_budgets[i] for i in I_act)
            N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)

            if H_sum > 0 and N_sum > 0:
                B_nom_t = (
                    pulp.lpSum(
                        self.data.weights_nominal.get((i, p, t), 1.0)
                        * self.data.counts.get((i, p, t), 0)
                        * B_nom_ip[(i, p, t)]
                        for i in I_act for p in P_t if self.data.counts.get((i, p, t), 0) > 0
                    )
                    / (H_sum * N_sum)
                )
            else:
                B_nom_t = 0

            Op_t = self.config.lambda_add * pulp.lpSum(
                self.config.get_add_cost(i, j, t) * a[(i, j, t)] for i in I for j in J
            )

            if self.config.use_robust and H_sum > 0 and N_sum > 0:
                B_wc_t = (
                    pulp.lpSum(
                        self.data.weights_nominal.get((i, p, t), 1.0)
                        * self.data.counts.get((i, p, t), 0)
                        * B_wc_ip[(i, p, t)]
                        for i in I_act for p in P_t if self.data.counts.get((i, p, t), 0) > 0
                    )
                    / (H_sum * N_sum)
                )
                obj_terms.append((1.0 - self.config.rho) * B_nom_t + self.config.rho * B_wc_t - Op_t)
            else:
                obj_terms.append(B_nom_t - Op_t)

        self.model += pulp.lpSum(obj_terms)

        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.config.solver_time_limit,
            gapRel=self.config.solver_gap,
            msg=0,
        )
        self.model.solve(solver)
        status = pulp.LpStatus[self.model.status]

        # Build per-slot solution objects for compatibility with existing reporting
        A_prev = self.data.initial_state.copy()
        for t in T:
            sol = SlotSolution(time_slot=t, status=status)

            if status in ["Optimal", "Feasible"]:
                for i in I:
                    for j in J:
                        if pulp.value(a[(i, j, t)]) and pulp.value(a[(i, j, t)]) > 0.5:
                            sol.adds[(i, j)] = 1
                        if pulp.value(r[(i, j, t)]) and pulp.value(r[(i, j, t)]) > 0.5:
                            sol.removes[(i, j)] = 1
                        if pulp.value(x[(i, j, t)]) and pulp.value(x[(i, j, t)]) > 0.5:
                            sol.states[(i, j)] = 1

                breakdown = self.calculator.calculate(
                    t=t,
                    state_current=sol.states,
                    state_prev=A_prev,
                )
                sol.R_nominal = float(breakdown.R_nominal)
                sol.B_nominal = float(breakdown.B_nominal)
                sol.Op_cost = float(breakdown.Op_cost)
                sol.R_wc = float(breakdown.R_wc) if self.config.use_robust else 0.0
                sol.B_wc = float(breakdown.B_wc) if self.config.use_robust else 0.0
                sol.objective_value = float(
                    breakdown.objective_total if self.config.use_robust else breakdown.objective_nominal
                )

                A_prev = sol.states.copy()

            self.solution.add_slot_solution(sol)

        # Export trace CSV in the same format as online solvers
        trace_path = self._trace_csv_path()
        with trace_path.open("w", newline="", encoding="utf-8") as trace_file:
            trace_writer = csv.DictWriter(trace_file, fieldnames=[
                "slot", "timestamp", "action_count", "adds", "removes", "actions",
                "placement_before", "placement_after", "demand", "attachment_points",
                "total_requests", "demand_pairs", "covered_pairs_after", "uncovered_pairs_after",
                "coverage_ratio_after", "objective", "R_nominal", "B_nominal", "R_wc", "B_wc",
                "Op_cost", "solve_time", "status",
            ])
            trace_writer.writeheader()

            A_prev = self.data.initial_state.copy()
            for sol in self.solution.slot_solutions:
                trace_writer.writerow(self._build_trace_row(sol.time_slot, A_prev, sol))
                A_prev = sol.states.copy()

        self.solution.trace_csv_path = str(trace_path)
        return self.solution
