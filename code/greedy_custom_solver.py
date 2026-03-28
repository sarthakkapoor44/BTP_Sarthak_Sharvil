from solver_base import BaseSolver,SlotSolution
from objective_calculator import ObjectiveCalculator
from typing import Dict, Tuple, List
import numpy as np
import math, time

class GreedyCustomSolver(BaseSolver):
    """
    Greedy baseline aligned numerically with the MILP:
      • R_nominal normalization: sum_i sum_j size[i]*removed(i,j) / sum_i sum_j size[i]*A_prev(i,j)  (active i only)
      • B_nominal normalization: sum_{i,p} w_bar(i,p,t)*n(i,p,t)*benefit(i,p) / [ (sum_i H_i over active i) * (sum_{i,p} n(i,p,t)) ]
        where benefit(i,p) = max(0, H_i - min_hops_from_p_to_any_server_storing_i)
      • Objective: (1 - rho) * (R_nominal + B_nominal) - lambda_add * Op_cost
    Notes:
      • APs are indexed in P_t and share index-space with servers for hop_distances[(j,p)].
      • Op_cost counts only prev=0 -> now=1 adds, using config.get_add_cost(i,j,t) if present.
    """

    def __init__(self, config, data):
        super().__init__(config, data)
        self.calculator = ObjectiveCalculator(config, data)

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        tic = time.time()

        I = self.config.num_datasets
        J = self.config.num_servers

        rho = float(getattr(self.config, "rho", 0.0))
        lambda_add = float(getattr(self.config, "lambda_add", 0.0))

        # --- knobs / matrices ---
        # We'll still keep a "reachability" cap for greedy feasibility checks
        # but the benefit uses per-dataset hop budgets like the MILP.
        max_hops_for_feas = int(getattr(self.config, "greedy_max_hops", 1))

        # Build all-pairs (server->server/AP) hop matrix from hop_distances
        shortest_paths = getattr(self.config, "shortest_paths", None)
        if shortest_paths is None:
            shortest_paths = np.full((J, J), np.inf, dtype=float)
            for j in range(J):
                shortest_paths[j, j] = 0.0
            hop_d = getattr(self.config, "hop_distances", {})
            for (jj, kk), d in hop_d.items():
                try:
                    d_val = float(d)
                except Exception:
                    d_val = math.inf
                shortest_paths[int(jj), int(kk)] = d_val

        # ---------- initial placement from A_prev ----------
        initial_storage = np.zeros((J, I), dtype=bool)
        for (i, j), v in A_prev.items():
            if v:
                initial_storage[j, i] = True

        # ---------- demand side (APs) ----------
        P_t: List[int] = self.data.attachment_points.get(t, [])

        # Also keep a request matrix (server-as-AP × dataset) only for feasibility checks
        request_vector = np.zeros((J, I), dtype=float)
        for p in P_t:
            k = int(p)
            for i in range(I):
                request_vector[k, i] += float(self.data.counts.get((i, p, t), 0))

        # ---------- greedy removal (same spirit) ----------
        current_storage = initial_storage.copy().astype(bool)
        removed_list: List[Tuple[int, int]] = []

        for i in range(I):
            servers_with_i = [j for j in range(J) if current_storage[j, i]]

            def removal_order_score(j: int):
                # Larger = try to remove earlier
                val = 0.0
                for x in range(J):
                    if request_vector[x, i] > 0:
                        val += shortest_paths[x, j]
                return val

            servers_sorted = sorted(range(J), key=lambda j: -removal_order_score(j))

            # Ensure all copies present first (as in your draft)
            for j in range(J):
                current_storage[j, i] = True

            # Try removals
            for j in servers_sorted:
                if not current_storage[j, i]:
                    continue

                current_storage[j, i] = False
                
                feasible = True
                for k in range(J):
                    if request_vector[k, i] <= 0:
                        continue
                    reachable = False
                    for j2 in range(J):
                        if current_storage[j2, i] and shortest_paths[k, j2] <= max_hops_for_feas:
                            reachable = True
                            break
                    if not reachable:
                        feasible = False
                        break

                if feasible:
                    if initial_storage[j, i]:
                        removed_list.append((i, j))
                else:
                    current_storage[j, i] = True

        # ================= Metrics aligned with MILP =================
        
        # Convert current_storage to state dict for calculator
        state_current = {(i, j): 1 if current_storage[j, i] else 0
                        for i in range(I) for j in range(J)}
        
        # Use centralized calculator for all objectives
        breakdown = self.calculator.calculate(
            t=t,
            state_current=state_current,
            state_prev=A_prev
        )
        
        # ================= Package solution =================
        sol = SlotSolution(time_slot=t, status="Feasible")
        
        # Derive Adds/Removes
        adds: Dict[Tuple[int, int], int] = {}
        removes: Dict[Tuple[int, int], int] = {}
        for i in range(I):
            for j in range(J):
                prev = 1 if initial_storage[j, i] else 0
                now = 1 if current_storage[j, i] else 0
                if prev == 0 and now == 1:
                    adds[(i, j)] = 1
                elif prev == 1 and now == 0:
                    removes[(i, j)] = 1
        
        sol.adds = adds
        sol.removes = removes
        sol.states = state_current

        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.R_wc = float(breakdown.R_wc)
        sol.B_wc = float(breakdown.B_wc)
        sol.Op_cost = float(breakdown.Op_cost)

        # Use robust or nominal objective based on config
        sol.objective_value = (breakdown.objective_total if self.config.use_robust 
                              else breakdown.objective_nominal)

        sol.solve_time = time.time() - tic
        return sol
