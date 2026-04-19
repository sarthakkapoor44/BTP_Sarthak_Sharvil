from solver_base import BaseSolver, SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator
from typing import Dict, Tuple, List
import itertools
import csv
import numpy as np
import os
from pathlib import Path


class OfflineMILPSolver(BaseSolver):
    """
    Strict offline global-horizon solver over the full horizon.

    One global first-stage model is solved for all slots t=1..T jointly.
    It is an oracle only when the global model is proven optimal.

    Modes:
    - Non-robust: optimize sum_t (R_nom_t + B_nom_t - Op_t)
    - Robust: optimize sum_t ((1-rho)*(R_nom_t + B_nom_t) + rho*eta_t - Op_t)
      with per-slot CCG cuts for eta_t.

    Notes:
    - Exact R_nom_t is enforced via bilinear equalities N_t = R_nom_t * D_t.
      This requires Gurobi (NonConvex=2).
    - Robust mode uses CCG over failure scenarios per slot, matching the online
      LaTeX structure but jointly optimized across all time slots.
    """

    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        super().__init__(config, data_gen)
        self.model = None
        self.calculator = ObjectiveCalculator(config, data_gen)

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        raise NotImplementedError("OfflineMILPSolver optimizes the global horizon via solve_all().")

    def _ensure_gurobi_license(self):
        """
        Force default use of workspace gurobi.lic unless GRB_LICENSE_FILE is
        already set by the user.
        """
        if os.environ.get("GRB_LICENSE_FILE"):
            return

        repo_root = Path(__file__).resolve().parent.parent
        lic_path = repo_root / "gurobi.lic"
        if lic_path.exists():
            os.environ["GRB_LICENSE_FILE"] = str(lic_path)

    @staticmethod
    def _bs_penalty_closed_form(vals: List[float], gamma: float) -> float:
        if not vals or gamma <= 0:
            return 0.0
        vals = sorted(vals, reverse=True)
        g_floor = int(np.floor(gamma))
        frac = gamma - g_floor
        s_val = sum(vals[:g_floor]) if g_floor > 0 else 0.0
        if frac > 1e-12 and g_floor < len(vals):
            s_val += frac * vals[g_floor]
        return s_val

    def _evaluate_recourse_max_given_f(
        self,
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        x_fixed: Dict[Tuple[int, int], int],
        f_fixed: Dict[int, int],
    ) -> Dict:
        import pulp

        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        model = pulp.LpProblem(f"OfflineRecourseMAX_t{t}", pulp.LpMaximize)

        y = pulp.LpVariable.dicts("y_fix", [(i, j) for i in I for j in J], lowBound=0, upBound=1)
        q = pulp.LpVariable.dicts("q_fix", [(i, j) for i in I for j in J], cat="Binary")
        z = pulp.LpVariable.dicts("z_fix", [(i, j) for i in I for j in J], cat="Binary")

        w_wc = {}
        B_wc_ip = {}
        for i in I:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                for d in range(0, H_i + 1):
                    w_wc[(i, p, d)] = pulp.LpVariable(f"w_wc_fix_{i}_{p}_{d}", cat="Binary")
                B_wc_ip[(i, p)] = pulp.LpVariable(
                    f"B_wc_ip_fix_{i}_{p}",
                    lowBound=0,
                    upBound=self.config.hop_budgets[i],
                )

        for i in I:
            for j in J:
                x_val = x_fixed.get((i, j), 0)
                f_val = f_fixed.get(j, 0)
                model += y[(i, j)] <= x_val
                model += y[(i, j)] <= 1 - f_val
                model += y[(i, j)] >= x_val - f_val

        for i in I:
            for j in J:
                model += q[(i, j)] <= y[(i, j)]
                model += z[(i, j)] == y[(i, j)] - q[(i, j)]

        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue
                f_p = f_fixed.get(p, 0)
                neighborhood = self.data.get_neighborhood(p, H_i)
                model += pulp.lpSum(z[(i, j)] for j in neighborhood) >= (1 - f_p)

        for i in I:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                count = self.data.counts.get((i, p, t), 0)
                if count <= 0:
                    for d in range(0, H_i + 1):
                        model += w_wc[(i, p, d)] <= 0
                    model += B_wc_ip[(i, p)] <= 0
                    continue

                f_p = f_fixed.get(p, 0)
                for d in range(0, H_i + 1):
                    ring_d = [j for j in J if self.config.hop_distances.get((j, p), float("inf")) <= d]
                    model += w_wc[(i, p, d)] <= pulp.lpSum(z[(i, j)] for j in ring_d)
                    model += w_wc[(i, p, d)] <= (1 - f_p)
                    if d > 0:
                        model += w_wc[(i, p, d - 1)] <= w_wc[(i, p, d)]

                model += B_wc_ip[(i, p)] <= pulp.lpSum(w_wc[(i, p, d)] for d in range(0, H_i + 1)) - (1 - f_p)
                model += B_wc_ip[(i, p)] >= 0
                model += B_wc_ip[(i, p)] <= H_i * (1 - f_p)

        denom_R = sum(
            self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
            for i in I for j in J
            if self.data.active_datasets.get((i, t), 0) == 1
        )
        if denom_R > 0:
            R_wc_expr = pulp.lpSum(
                self.config.dataset_sizes[i] * q[(i, j)]
                for i in I for j in J
                if self.data.active_datasets.get((i, t), 0) == 1
            ) / denom_R
        else:
            R_wc_expr = 0

        I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
        H_sum = sum(self.config.hop_budgets[i] for i in I_act)
        N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)
        P_alive = [p for p in P_t if f_fixed.get(p, 0) == 0]

        if H_sum > 0 and N_sum > 0 and P_alive:
            B_wc_nom_expr = pulp.lpSum(
                self.data.weights_nominal.get((i, p, t), 1.0)
                * self.data.counts.get((i, p, t), 0)
                * B_wc_ip[(i, p)]
                for i in I_act for p in P_alive if self.data.counts.get((i, p, t), 0) > 0
            ) / (H_sum * N_sum)
        else:
            B_wc_nom_expr = 0

        model += R_wc_expr + B_wc_nom_expr

        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.config.solver_time_limit,
            gapRel=self.config.solver_gap,
            msg=0,
        )
        model.solve(solver)

        status = pulp.LpStatus[model.status]
        if status not in ["Optimal", "Feasible"]:
            return {"R_wc_nom": 0.0, "B_wc_nom": 0.0, "B_wc_ip_vals": {}}

        R_wc_val = pulp.value(R_wc_expr) if isinstance(R_wc_expr, pulp.LpAffineExpression) else R_wc_expr
        B_wc_nom_val = pulp.value(B_wc_nom_expr) if isinstance(B_wc_nom_expr, pulp.LpAffineExpression) else B_wc_nom_expr
        B_vals = {}
        for i in I:
            for p in P_t:
                B_vals[(i, p)] = pulp.value(B_wc_ip[(i, p)]) or 0.0

        return {"R_wc_nom": float(R_wc_val), "B_wc_nom": float(B_wc_nom_val), "B_wc_ip_vals": B_vals}

    def _solve_adversarial_subproblem(
        self,
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        x_fixed: Dict[Tuple[int, int], int],
    ) -> Dict:
        I = range(self.config.num_datasets)
        J = list(range(self.config.num_servers))
        P_t = self.data.attachment_points.get(t, [])

        relevant_servers = set()
        for j in J:
            if j in P_t:
                relevant_servers.add(j)
            else:
                for i in I:
                    if x_fixed.get((i, j), 0) == 1:
                        relevant_servers.add(j)
                        break
        relevant_servers = list(relevant_servers)

        K_max = min(len(relevant_servers), self.config.K_failures)

        best_min_objective = float("inf")
        best_failure_set = ()
        best_R_wc = 0.0
        best_B_wc = 0.0

        for k in range(0, K_max + 1):
            for failure_combo in itertools.combinations(relevant_servers, k):
                f_current = {j: 0 for j in J}
                for failed_node in failure_combo:
                    f_current[failed_node] = 1

                rec_eval = self._evaluate_recourse_max_given_f(t, A_prev, x_fixed, f_current)
                R_wc_nom = rec_eval["R_wc_nom"]
                B_wc_nom = rec_eval["B_wc_nom"]
                B_wc_ip_vals = rec_eval["B_wc_ip_vals"]

                I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
                H_sum = sum(self.config.hop_budgets[i] for i in I_act)
                P_alive = [p for p in P_t if f_current.get(p, 0) == 0]

                penalty = 0.0
                if H_sum > 0 and P_alive:
                    N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)
                    if N_sum > 0:
                        pairs_vals = []
                        for i in I_act:
                            for p in P_alive:
                                n_ip = self.data.counts.get((i, p, t), 0)
                                if n_ip <= 0:
                                    continue
                                qcoef = (n_ip * B_wc_ip_vals.get((i, p), 0.0)) / (H_sum * N_sum)
                                hat = self.data.weights_error.get((i, p, t), 0.0)
                                if qcoef > 0 and hat > 0:
                                    pairs_vals.append(hat * qcoef)

                        penalty = self._bs_penalty_closed_form(pairs_vals, self.config.Gamma_budget)

                robust_B = B_wc_nom - penalty
                current_obj = R_wc_nom + robust_B

                if current_obj < best_min_objective:
                    best_min_objective = current_obj
                    best_failure_set = failure_combo
                    best_R_wc = R_wc_nom
                    best_B_wc = robust_B

        return {
            "failures": {j: 1 for j in best_failure_set},
            "objective": best_min_objective,
            "R_wc": best_R_wc,
            "B_wc": best_B_wc,
        }

    def _build_and_solve_master(self, scenarios_by_slot: Dict[int, List[Dict[int, int]]]):
        import importlib

        self._ensure_gurobi_license()

        try:
            gp = importlib.import_module("gurobipy")
            GRB = gp.GRB
        except ImportError as exc:
            raise RuntimeError("Global offline oracle requires gurobipy (Gurobi).") from exc

        I = list(range(self.config.num_datasets))
        J = list(range(self.config.num_servers))
        T = list(range(1, self.config.T + 1))

        model = gp.Model("uEDDE_offline_global")
        model.Params.OutputFlag = 1
        model.Params.TimeLimit = GRB.INFINITY
        # model.Params.MIPGap = float(self.config.solver_gap)
        model.Params.NonConvex = 2

        ijt = [(i, j, t) for i in I for j in J for t in T]
        a = model.addVars(ijt, vtype=GRB.BINARY, name="a")
        r = model.addVars(ijt, vtype=GRB.BINARY, name="r")
        x = model.addVars(ijt, vtype=GRB.BINARY, name="x")

        w_nom = {}
        B_nom_ip = {}
        for t in T:
            P_t = self.data.attachment_points.get(t, [])
            for i in I:
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    for d in range(0, H_i + 1):
                        w_nom[(i, p, d, t)] = model.addVar(vtype=GRB.BINARY, name=f"w_nom_{i}_{p}_{d}_{t}")
                    B_nom_ip[(i, p, t)] = model.addVar(
                        lb=0.0,
                        ub=float(H_i),
                        vtype=GRB.CONTINUOUS,
                        name=f"B_nom_{i}_{p}_{t}",
                    )

        R_nom_t = model.addVars(T, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="R_nom")
        numer_t = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="R_num")
        denom_t = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="R_den")
        denom_pos_t = model.addVars(T, vtype=GRB.BINARY, name="R_den_pos")

        eta_t = None
        if self.config.use_robust:
            eta_t = model.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="eta")

        for t in T:
            for i in I:
                alpha = self.data.active_datasets.get((i, t), 0)
                for j in J:
                    prev = self.data.initial_state.get((i, j), 0) if t == 1 else x[(i, j, t - 1)]

                    model.addConstr(x[(i, j, t)] == alpha * (prev - r[(i, j, t)]) + a[(i, j, t)])
                    model.addConstr(r[(i, j, t)] <= prev)
                    model.addConstr(a[(i, j, t)] <= alpha)
                    model.addConstr(a[(i, j, t)] <= 1 - prev)
                    model.addConstr(a[(i, j, t)] + r[(i, j, t)] <= 1)

            for j in J:
                model.addConstr(
                    gp.quicksum(self.config.dataset_sizes[i] * x[(i, j, t)] for i in I)
                    <= self.config.server_capacities[j]
                )
                if self.config.use_ingress_constraint:
                    model.addConstr(
                        gp.quicksum(self.config.dataset_sizes[i] * a[(i, j, t)] for i in I)
                        <= self.config.server_bandwidth[j]
                    )

            P_t = self.data.attachment_points.get(t, [])
            for i in I:
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    count = self.data.counts.get((i, p, t), 0)
                    L_bar = 1 if count >= 1 else 0
                    neighborhood = self.data.get_neighborhood(p, H_i)
                    if L_bar > 0:
                        model.addConstr(gp.quicksum(x[(i, j, t)] for j in neighborhood) >= L_bar)

                    for d in range(0, H_i + 1):
                        ring_d = [j for j in J if self.config.hop_distances.get((j, p), float("inf")) <= d]
                        model.addConstr(w_nom[(i, p, d, t)] <= gp.quicksum(x[(i, j, t)] for j in ring_d))
                        model.addConstr(w_nom[(i, p, d, t)] <= L_bar)
                        if d > 0:
                            model.addConstr(w_nom[(i, p, d - 1, t)] <= w_nom[(i, p, d, t)])

                    model.addConstr(
                        B_nom_ip[(i, p, t)]
                        <= gp.quicksum(w_nom[(i, p, d, t)] for d in range(0, H_i + 1)) - L_bar
                    )
                    model.addConstr(B_nom_ip[(i, p, t)] >= 0)

            I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
            prev_expr = gp.quicksum(
                self.config.dataset_sizes[i] * (self.data.initial_state.get((i, j), 0) if t == 1 else x[(i, j, t - 1)])
                for i in I_act for j in J
            )
            num_expr = gp.quicksum(self.config.dataset_sizes[i] * r[(i, j, t)] for i in I_act for j in J)

            model.addConstr(denom_t[t] == prev_expr)
            model.addConstr(numer_t[t] == num_expr)

            M_t = float(sum(self.config.dataset_sizes[i] * len(J) for i in I_act))
            eps = 1e-6
            if M_t <= 0:
                model.addConstr(denom_pos_t[t] == 0)
                model.addConstr(denom_t[t] == 0)
                model.addConstr(numer_t[t] == 0)
                model.addConstr(R_nom_t[t] == 0)
            else:
                model.addConstr(denom_t[t] <= M_t * denom_pos_t[t])
                model.addConstr(denom_t[t] >= eps * denom_pos_t[t])
                model.addConstr(numer_t[t] <= M_t * denom_pos_t[t])
                model.addConstr(R_nom_t[t] <= denom_pos_t[t])
                model.addQConstr(numer_t[t] == R_nom_t[t] * denom_t[t])

        if self.config.use_robust:
            for t in T:
                P_t = self.data.attachment_points.get(t, [])
                I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]

                for s_idx, f_s in enumerate(scenarios_by_slot[t]):
                    y_s = {}
                    q_s = {}
                    z_s = {}
                    for i in I:
                        for j in J:
                            y_s[(i, j)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_t{t}_s{s_idx}_{i}_{j}")
                            q_s[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"q_t{t}_s{s_idx}_{i}_{j}")
                            z_s[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_t{t}_s{s_idx}_{i}_{j}")

                            f_val = f_s.get(j, 0)
                            model.addConstr(y_s[(i, j)] <= x[(i, j, t)])
                            model.addConstr(y_s[(i, j)] <= 1 - f_val)
                            model.addConstr(y_s[(i, j)] >= x[(i, j, t)] - f_val)

                            model.addConstr(q_s[(i, j)] <= y_s[(i, j)])
                            model.addConstr(z_s[(i, j)] == y_s[(i, j)] - q_s[(i, j)])

                    for i in I:
                        if self.data.active_datasets.get((i, t), 0) != 1:
                            continue
                        H_i = self.config.hop_budgets[i]
                        for p in P_t:
                            if self.data.counts.get((i, p, t), 0) <= 0:
                                continue
                            neighborhood = self.data.get_neighborhood(p, H_i)
                            f_p = f_s.get(p, 0)
                            model.addConstr(gp.quicksum(z_s[(i, j)] for j in neighborhood) >= (1 - f_p))

                    w_wc_s = {}
                    B_wc_s = {}
                    for i in I:
                        H_i = self.config.hop_budgets[i]
                        for p in P_t:
                            count = self.data.counts.get((i, p, t), 0)
                            f_p = f_s.get(p, 0)

                            if count <= 0:
                                for d in range(0, H_i + 1):
                                    v = model.addVar(vtype=GRB.BINARY, name=f"w_wc_t{t}_s{s_idx}_{i}_{p}_{d}")
                                    w_wc_s[(i, p, d)] = v
                                    model.addConstr(v <= 0)
                                b = model.addVar(lb=0.0, ub=float(H_i), vtype=GRB.CONTINUOUS, name=f"B_wc_t{t}_s{s_idx}_{i}_{p}")
                                B_wc_s[(i, p)] = b
                                model.addConstr(b <= 0)
                                continue

                            for d in range(0, H_i + 1):
                                v = model.addVar(vtype=GRB.BINARY, name=f"w_wc_t{t}_s{s_idx}_{i}_{p}_{d}")
                                w_wc_s[(i, p, d)] = v
                                ring_d = [j for j in J if self.config.hop_distances.get((j, p), float("inf")) <= d]
                                model.addConstr(v <= gp.quicksum(z_s[(i, j)] for j in ring_d))
                                model.addConstr(v <= (1 - f_p))
                                if d > 0:
                                    model.addConstr(w_wc_s[(i, p, d - 1)] <= v)

                            b = model.addVar(lb=0.0, ub=float(H_i), vtype=GRB.CONTINUOUS, name=f"B_wc_t{t}_s{s_idx}_{i}_{p}")
                            B_wc_s[(i, p)] = b
                            model.addConstr(b <= gp.quicksum(w_wc_s[(i, p, d)] for d in range(0, H_i + 1)) - (1 - f_p))
                            model.addConstr(b >= 0)
                            model.addConstr(b <= H_i * (1 - f_p))

                    denom_R_expr = gp.quicksum(
                        self.config.dataset_sizes[i] * (self.data.initial_state.get((i, j), 0) if t == 1 else x[(i, j, t - 1)])
                        for i in I for j in J
                        if self.data.active_datasets.get((i, t), 0) == 1
                    )
                    numer_wc_expr = gp.quicksum(
                        self.config.dataset_sizes[i] * q_s[(i, j)]
                        for i in I for j in J
                        if self.data.active_datasets.get((i, t), 0) == 1
                    )

                    denom_wc = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"den_wc_t{t}_s{s_idx}")
                    R_wc_s = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"R_wc_t{t}_s{s_idx}")
                    den_wc_pos = model.addVar(vtype=GRB.BINARY, name=f"den_wc_pos_t{t}_s{s_idx}")

                    model.addConstr(denom_wc == denom_R_expr)
                    M_wc = float(sum(self.config.dataset_sizes[i] * len(J) for i in I_act))
                    eps_wc = 1e-6
                    if M_wc <= 0:
                        model.addConstr(den_wc_pos == 0)
                        model.addConstr(denom_wc == 0)
                        model.addConstr(R_wc_s == 0)
                    else:
                        model.addConstr(denom_wc <= M_wc * den_wc_pos)
                        model.addConstr(denom_wc >= eps_wc * den_wc_pos)
                        model.addConstr(numer_wc_expr <= M_wc * den_wc_pos)
                        model.addConstr(R_wc_s <= den_wc_pos)
                        model.addQConstr(numer_wc_expr == R_wc_s * denom_wc)

                    H_sum = sum(self.config.hop_budgets[i] for i in I_act)
                    N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)
                    P_alive_s = [p for p in P_t if f_s.get(p, 0) == 0]
                    active_pairs = [
                        (i, p) for i in I_act for p in P_alive_s
                        if self.data.counts.get((i, p, t), 0) > 0
                    ]

                    if H_sum > 0 and N_sum > 0 and active_pairs:
                        B_wc_nominal = gp.quicksum(
                            self.data.weights_nominal.get((i, p, t), 1.0)
                            * self.data.counts.get((i, p, t), 0)
                            * B_wc_s[(i, p)]
                            for (i, p) in active_pairs
                        ) / float(H_sum * N_sum)

                        pi_s = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"pi_t{t}_s{s_idx}")
                        phi_s = {
                            (i, p): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"phi_t{t}_s{s_idx}_{i}_{p}")
                            for (i, p) in active_pairs
                        }

                        penalty = self.config.Gamma_budget * pi_s + gp.quicksum(phi_s[(i, p)] for (i, p) in active_pairs)
                        B_wc_expr = B_wc_nominal - penalty

                        for (i, p) in active_pairs:
                            qcoef = (self.data.counts.get((i, p, t), 0) * B_wc_s[(i, p)]) / float(H_sum * N_sum)
                            hat = self.data.weights_error.get((i, p, t), 0.0)
                            model.addConstr(phi_s[(i, p)] >= hat * qcoef - pi_s)
                    else:
                        B_wc_expr = 0.0

                    model.addConstr(eta_t[t] <= R_wc_s + B_wc_expr)

        obj_terms = []
        for t in T:
            P_t = self.data.attachment_points.get(t, [])
            I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]

            H_sum = sum(self.config.hop_budgets[i] for i in I_act)
            N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)

            if H_sum > 0 and N_sum > 0:
                B_nom_t = gp.quicksum(
                    self.data.weights_nominal.get((i, p, t), 1.0)
                    * self.data.counts.get((i, p, t), 0)
                    * B_nom_ip[(i, p, t)]
                    for i in I_act for p in P_t if self.data.counts.get((i, p, t), 0) > 0
                ) / float(H_sum * N_sum)
            else:
                B_nom_t = 0.0

            Op_t = self.config.lambda_add * gp.quicksum(
                self.config.get_add_cost(i, j, t) * a[(i, j, t)] for i in I for j in J
            )

            if self.config.use_robust:
                obj_terms.append((1.0 - self.config.rho) * (R_nom_t[t] + B_nom_t) + self.config.rho * eta_t[t] - Op_t)
            else:
                obj_terms.append((R_nom_t[t] + B_nom_t) - Op_t)

        model.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)
        model.optimize()

        self.model = model
        return model, a, r, x, GRB, T, I, J

    def solve_all(self):
        T = list(range(1, self.config.T + 1))
        J = list(range(self.config.num_servers))

        scenarios_by_slot = {t: [{j: 0 for j in J}] for t in T}
        final_pack = None

        max_iter = self.config.ccg_max_iterations if self.config.use_robust else 1

        for _ in range(max_iter):
            final_pack = self._build_and_solve_master(scenarios_by_slot)

            model, a, r, x, GRB, T_vals, I_vals, J_vals = final_pack

            if model.Status not in [GRB.OPTIMAL] and model.SolCount <= 0:
                print(f"[OfflineMILPSolver] Master problem solve ended with status {model.Status} and no solutions; terminating.")
                break

            if not self.config.use_robust:
                break

            x_fixed_all = {(i, j, t): (1 if x[(i, j, t)].X > 0.5 else 0) for i in I_vals for j in J_vals for t in T_vals}
            eta_vals = {t: float(model.getVarByName(f"eta[{t}]").X) for t in T_vals}

            added_any = False
            for t in T_vals:
                A_prev_t = {}
                for i in I_vals:
                    for j in J_vals:
                        if t == 1:
                            A_prev_t[(i, j)] = self.data.initial_state.get((i, j), 0)
                        else:
                            A_prev_t[(i, j)] = x_fixed_all[(i, j, t - 1)]

                x_t = {(i, j): x_fixed_all[(i, j, t)] for i in I_vals for j in J_vals}
                adv = self._solve_adversarial_subproblem(t, A_prev_t, x_t)

                if adv["objective"] < eta_vals[t] - self.config.ccg_tolerance:
                    signature = tuple(adv["failures"].get(j, 0) for j in J_vals)
                    existing = {
                        tuple(s.get(j, 0) for j in J_vals)
                        for s in scenarios_by_slot[t]
                    }
                    if signature not in existing:
                        scenarios_by_slot[t].append(adv["failures"].copy())
                        added_any = True

            if not added_any:
                break

        if final_pack is None:
            return self.solution

        model, a, r, x, GRB, T_vals, I_vals, J_vals = final_pack

        if model.Status == GRB.OPTIMAL:
            status = "Optimal"
        elif model.SolCount > 0:
            gap = getattr(model, "MIPGap", None)
            if isinstance(gap, (float, int)):
                status = f"Feasible(status={model.Status},gap={float(gap):.6f})"
            else:
                status = f"Feasible(status={model.Status})"
        else:
            status = f"Infeasible(status={model.Status})"

        if self.config.verbose and model.Status != GRB.OPTIMAL:
            try:
                obj_val = float(model.ObjVal) if model.SolCount > 0 else float("nan")
            except Exception:
                obj_val = float("nan")
            try:
                obj_bound = float(model.ObjBound)
            except Exception:
                obj_bound = float("nan")
            try:
                mip_gap = float(model.MIPGap) if model.SolCount > 0 else float("nan")
            except Exception:
                mip_gap = float("nan")

            print(
                "[OfflineMILPSolver] Global solve not proven optimal; "
                f"status={model.Status}, obj={obj_val:.6f}, bound={obj_bound:.6f}, gap={mip_gap:.6f}."
            )

        A_prev = self.data.initial_state.copy()
        for t in T_vals:
            sol = SlotSolution(time_slot=t, status=status)

            if status in ["Optimal", "Feasible"]:
                for i in I_vals:
                    for j in J_vals:
                        if a[(i, j, t)].X > 0.5:
                            sol.adds[(i, j)] = 1
                        if r[(i, j, t)].X > 0.5:
                            sol.removes[(i, j)] = 1
                        if x[(i, j, t)].X > 0.5:
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
                if sol.status in ["Optimal", "Feasible"]:
                    A_prev = sol.states.copy()

        self.solution.trace_csv_path = str(trace_path)
        return self.solution
