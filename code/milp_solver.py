from solver_base import BaseSolver,SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator
from typing import Dict,Tuple,List
import itertools
import pulp
import time
import numpy as np

class MILPSolver(BaseSolver):
    """
    MILP solver using PuLP with Column-and-Constraint Generation (CCG).
    Two-stage robust per LaTeX (Option B: rings d=0..H_i) with AP-alive gating:
      - If AP p fails in a scenario: its coverage/rings/benefit terms are dropped.

    Notes / Fixes vs previous version:
      • Adversary still knocks out nodes from J (servers). APs are assumed to be
        a subset of J (as the rest of the code gates with f[p] where p ∈ J).
      • All B-normalizations in scenario/recourse use ONLY *alive* APs in the
        denominator (N_sum_alive) to avoid washing out B and producing zeros.
      • η is now unbounded below (lowBound=None) to be a true epigraph variable.
      • More informative oracle logging.
    """

    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        super().__init__(config, data_gen)
        self.model = None
        self.vars = {}
        self.calculator = ObjectiveCalculator(config, data_gen)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        start_time = time.time()
        if self.config.use_robust and self.config.use_ccg:
            sol = self._solve_slot_ccg(t, A_prev)
        else:
            sol = self._solve_slot_direct(t, A_prev)
        sol.solve_time = time.time() - start_time
        return sol

    def _finalize_slot_objective(
        self,
        sol: SlotSolution,
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        robust_override: Dict = None,
    ):
        """Centralized objective/metric assignment for MILP slot outputs."""
        breakdown = self.calculator.calculate(
            t=t,
            state_current=sol.states,
            state_prev=A_prev,
        )

        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.Op_cost = float(breakdown.Op_cost)

        if self.config.use_robust:
            if robust_override is not None:
                sol.R_wc = float(robust_override.get("R_wc", breakdown.R_wc))
                sol.B_wc = float(robust_override.get("B_wc", breakdown.B_wc))
            else:
                sol.R_wc = float(breakdown.R_wc)
                sol.B_wc = float(breakdown.B_wc)

            sol.objective_value = (
                (1.0 - self.config.rho) * (sol.R_nominal + sol.B_nominal)
                + self.config.rho * (sol.R_wc + sol.B_wc)
                - sol.Op_cost
            )
        else:
            sol.R_wc = 0.0
            sol.B_wc = 0.0
            sol.objective_value = float(breakdown.objective_nominal)

    # -------------------------------------------------------------------------
    # CCG Adversarial: pick failures only (coverage separator + recourse eval)
    # -------------------------------------------------------------------------

    def _solve_adversarial_subproblem(self, t: int, A_prev: Dict[Tuple[int, int], int],
                                  x_fixed: Dict[Tuple[int, int], int]) -> Dict:
  
        I = range(self.config.num_datasets)
        J = list(range(self.config.num_servers))
        P_t = self.data.attachment_points.get(t, [])

        # Only consider failing APs or servers that actually host something in x_fixed
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

        best_min_objective = float('inf')
        best_failure_set = ()
        best_R_wc = 0.0
        best_B_wc = 0.0

        # IMPORTANT: iterate k = 0..K (≤ K), not only exactly K
        for k in range(0, K_max + 1):
            for failure_combo in itertools.combinations(relevant_servers, k):

                # Construct failure vector for this combo
                f_current = {j: 0 for j in J}
                for failed_node in failure_combo:
                    f_current[failed_node] = 1

                # 1) Recourse MAX (on survivors, AP-gated) for this failure pattern
                rec_eval = self._evaluate_recourse_max_given_f(t, A_prev, x_fixed, f_current)
                R_wc_nom = rec_eval["R_wc_nom"]   # normalized by same denom as master
                B_wc_nom = rec_eval["B_wc_nom"]   # sum(barUpsilon * n_ip * B_ip) / (H_sum * N_sum_alive)
                B_wc_ip_vals = rec_eval["B_wc_ip_vals"]  # raw per-(i,p) B_ip values

                # 2) Closed-form BS penalty (one-sided, AP-alive gated, same denom)
                I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
                H_sum = sum(self.config.hop_budgets[i] for i in I_act)
                P_alive = [p for p in P_t if f_current.get(p, 0) == 0]

                penalty = 0.0
                if H_sum > 0 and P_alive:
                    N_sum_alive = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_alive)
                    if N_sum_alive > 0:
                        pairs_vals = []
                        for i in I_act:
                            for p in P_alive:
                                n_ip = self.data.counts.get((i, p, t), 0)
                                if n_ip <= 0:
                                    continue
                                # qcoef = ((1 - f_p) * n_ip * B_ip) / (H_sum * N_sum_alive); here f_p=0 for alive
                                qcoef = (n_ip * B_wc_ip_vals.get((i, p), 0.0)) / (H_sum * N_sum_alive)
                                hat = self.data.weights_error.get((i, p, t), 0.0)
                                if qcoef > 0 and hat > 0:
                                    pairs_vals.append(hat * qcoef)

                        # Sum of largest floor(Gamma) values + fractional part
                        penalty = self._bs_penalty_closed_form(pairs_vals, self.config.Gamma_budget)

                # DO NOT clamp to 0; master doesn't clamp either
                robust_B = B_wc_nom - penalty

                # Adversary minimizes our (recourse-maximized) R_wc + robust_B
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
            "B_wc": best_B_wc
        }


    @staticmethod 
    def _bs_penalty_closed_form(vals: List[float], Gamma: float) -> float:
        if not vals or Gamma <= 0:
            return 0.0
        vals = sorted(vals, reverse=True)
        g = int(np.floor(Gamma))
        frac = Gamma - g
        s = sum(vals[:g]) if g > 0 else 0.0
        if frac > 1e-12 and g < len(vals):
            s += frac * vals[g]
        return s

    # -------------------------------------------------------------------------
    # Recourse-only maximization for fixed x and f  (AP-gated)
    # -------------------------------------------------------------------------
    def _evaluate_recourse_max_given_f(self, t: int, A_prev: Dict[Tuple[int, int], int],
                                       x_fixed: Dict[Tuple[int, int], int],
                                       f_fixed: Dict[int, int]) -> Dict:
        """
        MAX over recourse (q,z) (and y linkage) of:
            R_wc_nominal(q) + B_wc_nominal(z)
        given x and f fixed; subject to Survive, Rec, Cov-WC-Alive, Ring-w-Alive, B-def-Alive.
        """
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        model = pulp.LpProblem(f"RecourseMAX_t{t}", pulp.LpMaximize)

        # y, q, z
        y = pulp.LpVariable.dicts("y_fix", [(i, j) for i in I for j in J], lowBound=0, upBound=1)
        q = pulp.LpVariable.dicts("q_fix", [(i, j) for i in I for j in J], cat='Binary')
        z = pulp.LpVariable.dicts("z_fix", [(i, j) for i in I for j in J], cat='Binary')

        # Ring-w and B_wc_ip
        w_wc = {}
        B_wc_ip = {}
        for i in I:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                for d in range(0, H_i + 1):
                    w_wc[(i, p, d)] = pulp.LpVariable(f"w_wc_fix_{i}_{p}_{d}", cat='Binary')
                B_wc_ip[(i, p)] = pulp.LpVariable(f"B_wc_ip_fix_{i}_{p}", lowBound=0,
                                                  upBound=self.config.hop_budgets[i])

        # Survivor linkage y = x * (1 - f)
        for i in I:
            for j in J:
                x_val = x_fixed.get((i, j), 0)
                f_val = f_fixed.get(j, 0)
                model += y[(i, j)] <= x_val
                model += y[(i, j)] <= 1 - f_val
                model += y[(i, j)] >= x_val - f_val

        # Recourse constraints
        for i in I:
            for j in J:
                model += q[(i, j)] <= y[(i, j)]
                model += z[(i, j)] == y[(i, j)] - q[(i, j)]
                model += q[(i, j)] <= A_prev.get((i, j), 0)

        # Post-recourse coverage (AP-alive gating)
        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue
                L_bar = 1
                f_p = f_fixed.get(p, 0)
                if L_bar > 0:
                    neighborhood = self.data.get_neighborhood(p, H_i)
                    # sum z >= L_bar*(1 - f_p)
                    model += pulp.lpSum(z[(i, j)] for j in neighborhood) >= L_bar * (1 - f_p)

        # Ring-w (AP-alive gating) + B-def-Alive
        for i in I:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                count = self.data.counts.get((i, p, t), 0)
                if count <= 0:
                    # Force these to 0 to tighten
                    for d in range(0, H_i + 1):
                        model += w_wc[(i, p, d)] <= 0
                    model += B_wc_ip[(i, p)] <= 0
                    continue

                f_p = f_fixed.get(p, 0)
                L_bar = 1
                for d in range(0, H_i + 1):
                    ring_d = [j for j in J if self.config.hop_distances.get((j, p), float('inf')) <= d]
                    model += w_wc[(i, p, d)] <= pulp.lpSum(z[(i, j)] for j in ring_d)
                    model += w_wc[(i, p, d)] <= L_bar * (1 - f_p)    # AP-alive cap
                    model += w_wc[(i, p, d)] <= (1 - f_p)            # redundant but safe
                    if d > 0:
                        model += w_wc[(i, p, d - 1)] <= w_wc[(i, p, d)]

                # 0 <= B_wc_ip <= H_i*(1 - f_p) and defn w.r.t. rings minus L_bar*(1 - f_p)
                model += B_wc_ip[(i, p)] <= pulp.lpSum(w_wc[(i, p, d)] for d in range(0, H_i + 1)) - L_bar * (1 - f_p)
                model += B_wc_ip[(i, p)] >= 0
                model += B_wc_ip[(i, p)] <= H_i * (1 - f_p)

        # Objective: R_wc_nominal + B_wc_nominal
        denom_R = sum(self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                      for i in I for j in J if self.data.active_datasets.get((i, t), 0) == 1)
        if denom_R > 0:
            R_wc_expr = pulp.lpSum(self.config.dataset_sizes[i] * q[(i, j)]
                                   for i in I for j in J
                                   if self.data.active_datasets.get((i, t), 0) == 1) / denom_R
        else:
            R_wc_expr = 0

        # N_sum over ALIVE APs only
        I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
        H_sum = sum(self.config.hop_budgets[i] for i in I_act)
        P_alive = [p for p in P_t if f_fixed.get(p, 0) == 0]
        N_sum_alive = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_alive)

        if H_sum > 0 and N_sum_alive > 0 and P_alive:
            # Note: B_wc_ip already gated by (1 - f_p), so nominal average is over alive APs.
            B_wc_nom_expr = pulp.lpSum(
                self.data.weights_nominal.get((i, p, t), 1.0) *
                self.data.counts.get((i, p, t), 0) * B_wc_ip[(i, p)]
                for i in I_act for p in P_alive if self.data.counts.get((i, p, t), 0) > 0
            ) / (H_sum * N_sum_alive)
        else:
            B_wc_nom_expr = 0

        model += R_wc_expr + B_wc_nom_expr

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=self.config.solver_time_limit,
                                   gapRel=self.config.solver_gap, msg=0)
        model.solve(solver)
        status = pulp.LpStatus[model.status]
        if status not in ['Optimal', 'Feasible']:
            return {"R_wc_nom": 0.0, "B_wc_nom": 0.0, "B_wc_ip_vals": {}}

        R_wc_val = pulp.value(R_wc_expr) if isinstance(R_wc_expr, pulp.LpAffineExpression) else R_wc_expr
        B_wc_nom_val = pulp.value(B_wc_nom_expr) if isinstance(B_wc_nom_expr, pulp.LpAffineExpression) else B_wc_nom_expr
        B_vals = {}
        for i in I:
            for p in P_t:
                if (i, p) in B_wc_ip:
                    B_vals[(i, p)] = pulp.value(B_wc_ip[(i, p)]) or 0.0

        return {"R_wc_nom": R_wc_val, "B_wc_nom": B_wc_nom_val, "B_wc_ip_vals": B_vals}

    # -------------------------------------------------------------------------
    # Main CCG Loop (AP-gated cuts)
    # -------------------------------------------------------------------------
    def _solve_slot_ccg(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        print(f"  Using CCG (iterative robust)...")
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        scenarios = [{"failures": {j: 0 for j in J}}]
        scenario_signatures = {tuple(0 for _ in J)}

        best_solution = None
        best_master_obj = -float('inf')

        def capture_solution(iter_count, status_label, adv_info):
            sol = SlotSolution(time_slot=t, status=status_label)
            sol.ccg_iterations = iter_count
            for i in I:
                for j in J:
                    if pulp.value(a[(i, j)]) > 0.5:
                        sol.adds[(i, j)] = 1
                    if pulp.value(r[(i, j)]) > 0.5:
                        sol.removes[(i, j)] = 1
                    if pulp.value(x[(i, j)]) > 0.5:
                        sol.states[(i, j)] = 1
            if adv_info is not None:
                sol.failures = adv_info.get('failures', {})
            for key, var in B_nom_ip.items():
                sol.benefit_nominal_per_ip[key] = pulp.value(var)
            self._finalize_slot_objective(sol, t, A_prev, robust_override=adv_info)
            return sol

        for iteration in range(self.config.ccg_max_iterations):
            print(f"    CCG iteration {iteration+1}/{self.config.ccg_max_iterations}")

            # ---------------------- MASTER ----------------------
            print(f"      Building master problem ({len(scenarios)} scenarios)...")
            master = pulp.LpProblem(f"Master_t{t}_iter{iteration}", pulp.LpMaximize)

            a = pulp.LpVariable.dicts("a", [(i, j) for i in I for j in J], cat='Binary')
            r = pulp.LpVariable.dicts("r", [(i, j) for i in I for j in J], cat='Binary')
            x = pulp.LpVariable.dicts("x", [(i, j) for i in I for j in J], cat='Binary')

            w_nom = {}
            for i in I:
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    for d in range(0, H_i + 1):
                        w_nom[(i, p, d)] = pulp.LpVariable(f"w_nom_{i}_{p}_{d}", cat='Binary')

            B_nom_ip = {}
            for i in I:
                for p in P_t:
                    B_nom_ip[(i, p)] = pulp.LpVariable(f"B_nom_{i}_{p}", lowBound=0, upBound=self.config.hop_budgets[i])

            eta = pulp.LpVariable("eta", lowBound=None)  # epigraph for Z(x), allow negative

            # (D1)-(D2) + XOR
            for i in I:
                for j in J:
                    alpha = self.data.active_datasets.get((i, t), 0)
                    A_prev_ij = A_prev.get((i, j), 0)
                    master += x[(i, j)] == alpha * (A_prev_ij - r[(i, j)]) + a[(i, j)]
                    master += r[(i, j)] <= A_prev_ij
                    master += a[(i, j)] <= alpha
                    master += a[(i, j)] <= 1 - A_prev_ij
                    master += a[(i, j)] + r[(i, j)] <= 1

            # Capacity + optional ingress
            for j in J:
                master += pulp.lpSum(self.config.dataset_sizes[i] * x[(i, j)] for i in I) <= self.config.server_capacities[j]
                if self.config.use_ingress_constraint:
                    master += pulp.lpSum(self.config.dataset_sizes[i] * a[(i, j)] for i in I) <= self.config.server_bandwidth[j]

            # Nominal coverage, rings, B-def
            for i in I:
                for p in P_t:
                    count = self.data.counts.get((i, p, t), 0)
                    L_bar = 1 if count >= 1 else 0
                    if L_bar > 0:
                        neighborhood = self.data.get_neighborhood(p, self.config.hop_budgets[i])
                        multiplier = 1
                        if getattr(self.config, "use_k_survivable_coverage", False):
                            multiplier = min(self.config.K_failures + 1, len(neighborhood))
                        master += pulp.lpSum(x[(i, j)] for j in neighborhood) >= multiplier * L_bar
                    H_i = self.config.hop_budgets[i]
                    for d in range(0, H_i + 1):
                        ring_d = [j for j in J if self.config.hop_distances.get((j, p), float('inf')) <= d]
                        master += w_nom[(i, p, d)] <= pulp.lpSum(x[(i, j)] for j in ring_d)
                        master += w_nom[(i, p, d)] <= L_bar
                        if d > 0:
                            master += w_nom[(i, p, d - 1)] <= w_nom[(i, p, d)]
                    master += B_nom_ip[(i, p)] <= pulp.lpSum(w_nom[(i, p, d)] for d in range(0, H_i + 1)) - L_bar
                    master += B_nom_ip[(i, p)] >= 0

            # Per-scenario recourse copies + BS duals -> eta cuts (AP-gated)
            for s_idx, scenario in enumerate(scenarios):
                f_s = scenario["failures"]

                # Survivors y_s
                y_s = {}
                for i in I:
                    for j in J:
                        y_s[(i, j)] = pulp.LpVariable(f"y_s{s_idx}_{i}_{j}", lowBound=0, upBound=1)
                        f_val = f_s.get(j, 0)
                        master += y_s[(i, j)] <= x[(i, j)]
                        master += y_s[(i, j)] <= 1 - f_val
                        master += y_s[(i, j)] >= x[(i, j)] - f_val

                # q_s, z_s
                q_s = {}
                z_s = {}
                for i in I:
                    for j in J:
                        q_s[(i, j)] = pulp.LpVariable(f"q_s{s_idx}_{i}_{j}", cat='Binary')
                        z_s[(i, j)] = pulp.LpVariable(f"z_s{s_idx}_{i}_{j}", cat='Binary')
                        master += q_s[(i, j)] <= y_s[(i, j)]
                        master += z_s[(i, j)] == y_s[(i, j)] - q_s[(i, j)]
                        master += q_s[(i, j)] <= A_prev.get((i, j), 0)

                # Cov-WC-Alive: sum z >= 1 * (1 - f_p) for active (i,p) with demand
                for i in I:
                    if self.data.active_datasets.get((i, t), 0) != 1:
                        continue
                    H_i = self.config.hop_budgets[i]
                    for p in P_t:
                        if self.data.counts.get((i, p, t), 0) <= 0:
                            continue
                        f_p = f_s.get(p, 0)
                        neighborhood = self.data.get_neighborhood(p, H_i)
                        master += pulp.lpSum(z_s[(i, j)] for j in neighborhood) >= (1 - f_p)

                # Ring-w-Alive + B-def-Alive
                w_wc_s = {}
                B_wc_s = {}
                for i in I:
                    H_i = self.config.hop_budgets[i]
                    for p in P_t:
                        count = self.data.counts.get((i, p, t), 0)
                        f_p = f_s.get(p, 0)
                        if count <= 0:
                            # tighten to zero
                            for d in range(0, H_i + 1):
                                v = pulp.LpVariable(f"w_wc_s{s_idx}_{i}_{p}_{d}", cat='Binary')
                                w_wc_s[(i, p, d)] = v
                                master += v <= 0
                            b = pulp.LpVariable(f"B_wc_s{s_idx}_{i}_{p}", lowBound=0, upBound=self.config.hop_budgets[i])
                            B_wc_s[(i, p)] = b
                            master += b <= 0
                            continue

                        for d in range(0, H_i + 1):
                            w_wc_s[(i, p, d)] = pulp.LpVariable(f"w_wc_s{s_idx}_{i}_{p}_{d}", cat='Binary')
                            ring_d = [j for j in J if self.config.hop_distances.get((j, p), float('inf')) <= d]
                            master += w_wc_s[(i, p, d)] <= pulp.lpSum(z_s[(i, j)] for j in ring_d)
                            master += w_wc_s[(i, p, d)] <= (1 - f_p)
                            if d > 0:
                                master += w_wc_s[(i, p, d - 1)] <= w_wc_s[(i, p, d)]

                        B_wc_s[(i, p)] = pulp.LpVariable(f"B_wc_s{s_idx}_{i}_{p}", lowBound=0,
                                                          upBound=self.config.hop_budgets[i])
                        # B <= sum w - (1 - f_p)
                        master += B_wc_s[(i, p)] <= pulp.lpSum(w_wc_s[(i, p, d)] for d in range(0, H_i + 1)) - (1 - f_p)
                        master += B_wc_s[(i, p)] >= 0
                        master += B_wc_s[(i, p)] <= H_i * (1 - f_p)

                # R_wc_s (same denom)
                denom_R = sum(self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                              for i in I for j in J if self.data.active_datasets.get((i, t), 0) == 1)
                if denom_R > 0:
                    R_wc_s = pulp.lpSum(self.config.dataset_sizes[i] * q_s[(i, j)]
                                        for i in I for j in J
                                        if self.data.active_datasets.get((i, t), 0) == 1) / denom_R
                else:
                    R_wc_s = 0

                # B_wc nominal - BS penalty (dualized, AP-gated) with alive denom
                I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
                H_sum = sum(self.config.hop_budgets[i] for i in I_act)
                P_alive_s = [p for p in P_t if f_s.get(p, 0) == 0]
                N_sum_alive = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_alive_s)
                active_pairs = [(i, p) for i in I_act for p in P_alive_s if self.data.counts.get((i, p, t), 0) > 0]

                if H_sum > 0 and N_sum_alive > 0 and active_pairs:
                    B_wc_nominal = pulp.lpSum(
                        self.data.weights_nominal.get((i, p, t), 1.0) *
                        self.data.counts.get((i, p, t), 0) * B_wc_s[(i, p)]
                        for (i, p) in active_pairs
                    ) / (H_sum * N_sum_alive)
                    pi_s = pulp.LpVariable(f"pi_s{s_idx}", lowBound=0)
                    phi_s = {(i, p): pulp.LpVariable(f"phi_s{s_idx}_{i}_{p}", lowBound=0) for (i, p) in active_pairs}
                    penalty = self.config.Gamma_budget * pi_s + pulp.lpSum(phi_s[(i, p)] for (i, p) in active_pairs)
                    B_wc_expr = B_wc_nominal - penalty
                    # BS dual constraints: phi >= hat * qcoef - pi, with qcoef AP-gated over alive denom
                    for (i, p) in active_pairs:
                        qcoef = (self.data.counts.get((i, p, t), 0) * B_wc_s[(i, p)]) / (H_sum * N_sum_alive)
                        hat = self.data.weights_error.get((i, p, t), 0.0)
                        master += phi_s[(i, p)] >= hat * qcoef - pi_s
                else:
                    B_wc_expr = 0

                # Eta cut
                master += eta <= R_wc_s + B_wc_expr

            # Master objective (nominal part stays global, not scenario-gated)
            denom_R = sum(self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                          for i in I for j in J if self.data.active_datasets.get((i, t), 0) == 1)
            if denom_R > 0:
                R_nom = pulp.lpSum(self.config.dataset_sizes[i] * r[(i, j)]
                                   for i in I for j in J
                                   if self.data.active_datasets.get((i, t), 0) == 1) / denom_R
            else:
                R_nom = 0

            H_sum_global = sum(self.config.hop_budgets[i] for i in I if self.data.active_datasets.get((i, t), 0) == 1)
            N_sum_global = sum(self.data.counts.get((i, p, t), 0) for i in I for p in P_t)
            if H_sum_global > 0 and N_sum_global > 0:
                B_nom = pulp.lpSum(
                    self.data.weights_nominal.get((i, p, t), 1.0) *
                    self.data.counts.get((i, p, t), 0) * B_nom_ip[(i, p)]
                    for i in I for p in P_t if self.data.counts.get((i, p, t), 0) > 0
                ) / (H_sum_global * N_sum_global)
            else:
                B_nom = 0

            Op = self.config.lambda_add * pulp.lpSum(self.config.get_add_cost(i, j, t) * a[(i, j)] for i in I for j in J)
            master_obj = (1 - self.config.rho) * (R_nom + B_nom) + self.config.rho * eta - Op
            master += master_obj

            # Solve master
            print("      Solving master problem...")
            solver = pulp.PULP_CBC_CMD(timeLimit=self.config.solver_time_limit,
                                       gapRel=self.config.solver_gap, msg=0)
            master.solve(solver)
            status = pulp.LpStatus[master.status]
            print(f"      Master status: {status}, obj={pulp.value(master.objective):.6f}")
            if status not in ['Optimal', 'Feasible']:
                return SlotSolution(time_slot=t, status='Infeasible')

            x_fixed = {(i, j): (1 if pulp.value(x[(i, j)]) > 0.5 else 0) for i in I for j in J}
            master_obj_val = pulp.value(master.objective)
            eta_val = pulp.value(eta)
            print(f"      Master: obj={master_obj_val:.6f}, eta={eta_val:.6f}")

            # ---------------------- ORACLE ----------------------
            adv_result = self._solve_adversarial_subproblem(t, A_prev, x_fixed)
            if adv_result is None:
                return SlotSolution(time_slot=t, status='Infeasible')

            adv_val = adv_result['objective']
            print(f"      Oracle: R_wc={adv_result['R_wc']:.8f}, B_wc={adv_result['B_wc']:.8f}, "
                  f"adv={adv_val:.8f}, eta={eta_val:.8f}, gap={(eta_val - adv_val):.6e}")

            # Convergence: oracle lower bound >= master η - tol
            if adv_val >= eta_val - self.config.ccg_tolerance:
                print("      ✓ Converged by robust lower bound >= eta (within tolerance).")
                candidate_solution = capture_solution(iteration + 1, 'Optimal', adv_result)
                return candidate_solution

            # Else add new scenario and continue
            candidate_solution = capture_solution(iteration + 1, 'Feasible', adv_result)
            if master_obj_val > best_master_obj:
                best_solution, best_master_obj = candidate_solution, master_obj_val

            signature = tuple(adv_result['failures'].get(j, 0) for j in J)
            if signature in scenario_signatures:
                print("      Scenario already present; returning best-known solution.")
                return best_solution if best_solution is not None else candidate_solution

            scenarios.append({'failures': adv_result['failures'].copy()})
            scenario_signatures.add(signature)
            print(f"      → Added scenario #{len(scenarios)} (failures: {sum(adv_result['failures'].values())})")

        print("    CCG: Max iterations reached.")
        return best_solution if best_solution is not None else SlotSolution(time_slot=t, status='Infeasible')

    # -------------------------------------------------------------------------
    # Direct (non-robust) solve. Robust direct is disabled to avoid min–max.
    # -------------------------------------------------------------------------
    def _solve_slot_direct(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        if self.config.use_robust:
            raise RuntimeError("Direct robust is unsupported; use CCG for min–max robust solving.")
        print("  Building nominal MILP model (direct)...")

        self.model = pulp.LpProblem(f"uEDDE_t{t}", pulp.LpMaximize)
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        a = pulp.LpVariable.dicts("a", [(i, j) for i in I for j in J], cat='Binary')
        r = pulp.LpVariable.dicts("r", [(i, j) for i in I for j in J], cat='Binary')
        x = pulp.LpVariable.dicts("x", [(i, j) for i in I for j in J], cat='Binary')

        w_nom = {}
        B_nom_ip = {}
        for i in I:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                for d in range(0, H_i + 1):
                    w_nom[(i, p, d)] = pulp.LpVariable(f"w_nom_{i}_{p}_{d}", cat='Binary')
                B_nom_ip[(i, p)] = pulp.LpVariable(f"B_nom_{i}_{p}", lowBound=0, upBound=self.config.hop_budgets[i])

        # (D1)-(D2) + XOR
        for i in I:
            for j in J:
                alpha = self.data.active_datasets.get((i, t), 0)
                A_prev_ij = A_prev.get((i, j), 0)
                self.model += x[(i, j)] == alpha * (A_prev_ij - r[(i, j)]) + a[(i, j)]
                self.model += r[(i, j)] <= A_prev_ij
                self.model += a[(i, j)] <= alpha
                self.model += a[(i, j)] <= 1 - A_prev_ij
                self.model += a[(i, j)] + r[(i, j)] <= 1

        # Capacity + optional ingress
        for j in J:
            self.model += pulp.lpSum(self.config.dataset_sizes[i] * x[(i, j)] for i in I) <= self.config.server_capacities[j]
            if self.config.use_ingress_constraint:
                self.model += pulp.lpSum(self.config.dataset_sizes[i] * a[(i, j)] for i in I) <= self.config.server_bandwidth[j]

        # Nominal coverage, rings, B-def
        for i in I:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                count = self.data.counts.get((i, p, t), 0)
                L_bar = 1 if count >= 1 else 0
                if L_bar > 0:
                    neighborhood = self.data.get_neighborhood(p, H_i)
                    self.model += pulp.lpSum(x[(i, j)] for j in neighborhood) >= L_bar
                for d in range(0, H_i + 1):
                    ring_d = [j for j in J if self.config.hop_distances.get((j, p), float('inf')) <= d]
                    self.model += w_nom[(i, p, d)] <= pulp.lpSum(x[(i, j)] for j in ring_d)
                    self.model += w_nom[(i, p, d)] <= L_bar
                    if d > 0:
                        self.model += w_nom[(i, p, d - 1)] <= w_nom[(i, p, d)]
                self.model += B_nom_ip[(i, p)] <= pulp.lpSum(w_nom[(i, p, d)] for d in range(0, H_i + 1)) - L_bar
                self.model += B_nom_ip[(i, p)] >= 0

        # Objective
        denom_R = sum(self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                      for i in I for j in J if self.data.active_datasets.get((i, t), 0) == 1)
        if denom_R > 0:
            R_nom = pulp.lpSum(self.config.dataset_sizes[i] * r[(i, j)]
                               for i in I for j in J
                               if self.data.active_datasets.get((i, t), 0) == 1) / denom_R
        else:
            R_nom = 0

        H_sum = sum(self.config.hop_budgets[i] for i in I if self.data.active_datasets.get((i, t), 0) == 1)
        N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I for p in P_t)
        if H_sum > 0 and N_sum > 0:
            B_nom = pulp.lpSum(
                self.data.weights_nominal.get((i, p, t), 1.0) *
                self.data.counts.get((i, p, t), 0) * B_nom_ip[(i, p)]
                for i in I for p in P_t if self.data.counts.get((i, p, t), 0) > 0
            ) / (H_sum * N_sum)
        else:
            B_nom = 0

        Op = self.config.lambda_add * pulp.lpSum(self.config.get_add_cost(i, j, t) * a[(i, j)] for i in I for j in J)
        # Non-robust direct objective should not be scaled by rho.
        obj = (R_nom + B_nom) - Op
        self.model += obj

        # Solve
        print("  Solving nominal MILP...")
        solver = pulp.PULP_CBC_CMD(timeLimit=self.config.solver_time_limit,
                                   gapRel=self.config.solver_gap, msg=0)
        self.model.solve(solver)
        status = pulp.LpStatus[self.model.status]

        # Extract
        sol = SlotSolution(time_slot=t, status=status)
        if status in ['Optimal', 'Feasible']:
            for i in I:
                for j in J:
                    if pulp.value(a[(i, j)]) > 0.5:
                        sol.adds[(i, j)] = 1
                    if pulp.value(r[(i, j)]) > 0.5:
                        sol.removes[(i, j)] = 1
                    if pulp.value(x[(i, j)]) > 0.5:
                        sol.states[(i, j)] = 1
            for i in I:
                for p in P_t:
                    if (i, p) in B_nom_ip:
                        sol.benefit_nominal_per_ip[(i, p)] = pulp.value(B_nom_ip[(i, p)])
            self._finalize_slot_objective(sol, t, A_prev)
        return sol

