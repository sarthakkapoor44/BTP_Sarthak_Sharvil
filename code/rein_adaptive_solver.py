import math
import time
from typing import Dict, Tuple, List, Optional
from solver_base import BaseSolver, SlotSolution
from objective_calculator import ObjectiveCalculator

class REINAdaptiveSolver(BaseSolver):
    """
    Fixed REIN Solver using Adaptive Marginal Greedy.
    
    Why this fixes the results:
    Robust benefit under K failures is a STEP FUNCTION.
    - 1 replica: 0 robust benefit (Adversary kills it)
    - K replicas: 0 robust benefit (Adversary kills all)
    - K+1 replicas: HIGH robust benefit (1 survives)
    
    The previous 'modular' solver assumed linear benefit (1/(K+1)), leading to 
    sparse placements that got wiped out. This solver re-evaluates the 
    TRUE marginal gain at every step, forcing it to 'stack' replicas 
    until K+1 redundancy is reached.
    """

    def __init__(self, config, data):
        super().__init__(config, data)
        self.calculator = ObjectiveCalculator(config, data)

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        start_time = time.time()
        if getattr(self.config, "verbose", False):
            print(f"  > REIN-Adaptive: Solving slot {t} (K={self.config.K_failures}, rho={self.config.rho})...")
        
        # 1. Setup
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        K = max(0, int(getattr(self.config, "K_failures", 0)))
        rho = float(getattr(self.config, "rho", 0.0))
        active_APs = self.data.attachment_points.get(t, [])

        # Start with previous state (A_prev) as the base for stability
        current_state = A_prev.copy()
        
        # Track current capacity usage
        current_load = {j: 0.0 for j in J}
        for (i, j), val in current_state.items():
            if val == 1:
                current_load[j] += self.config.dataset_sizes[i]

        # 2. Adaptive Greedy Loop
        # We keep adding replicas as long as they provide positive marginal gain density
        iteration = 0
        while True:
            iteration += 1
            best_density = -float('inf')
            best_candidate = None
            best_gain_vals = None # Store (gain_nom, gain_rob, cost) for logging
            
            # Identify valid candidates: active datasets, not yet placed, fitting capacity
            candidates = []
            for i in I:
                if self.data.active_datasets.get((i, t), 0) != 1: 
                    continue
                size_i = self.config.dataset_sizes[i]
                
                for j in J:
                    # Skip if already placed
                    if current_state.get((i, j), 0) == 1: 
                        continue
                    # Skip if capacity full
                    if current_load[j] + size_i > self.config.server_capacities[j]:
                        continue
                    candidates.append((i, j))
            
            if not candidates:
                break

            # 3. Evaluate Marginal Gain for all candidates
            # This looks expensive, but we optimize inside _get_marginal_gain
            found_improvement = False
            
            for (i, j) in candidates:
                # Calculate how much Objective increases if we add (i,j) to current_state
                gain, g_nom, g_rob, cost = self._get_marginal_gain(
                    i, j, t, current_state, K, rho, A_prev, active_APs
                )
                
                if gain <= 1e-9:
                    continue

                density = gain / self.config.dataset_sizes[i]
                
                if density > best_density:
                    best_density = density
                    best_candidate = (i, j)
                    best_gain_vals = (gain, g_nom, g_rob, cost)
                    found_improvement = True
            
            if not found_improvement:
                break
            
            # 4. Commit Best Move
            i_best, j_best = best_candidate
            current_state[(i_best, j_best)] = 1
            current_load[j_best] += self.config.dataset_sizes[i_best]
            
            # Optional verbose logging for debugging
            # if iteration % 10 == 0:
            #     print(f"    Iter {iteration}: Added ({i_best},{j_best}). Gain={best_gain_vals[0]:.4f}")

        # 5. Construct Solution Object
        sol = SlotSolution(time_slot=t, status="Feasible")
        sol.states = current_state
        
        # Derive Adds/Removes
        for i in I:
            for j in J:
                curr = current_state.get((i, j), 0)
                prev = A_prev.get((i, j), 0)
                sol.states[(i, j)] = curr
                if curr == 1 and prev == 0:
                    sol.adds[(i, j)] = 1
                elif curr == 0 and prev == 1:
                    sol.removes[(i, j)] = 1

        # 6. Compute Final Exact Objectives using centralized calculator
        breakdown = self.calculator.calculate(
            t=t,
            state_current=current_state,
            state_prev=A_prev
        )
        
        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.R_wc = float(breakdown.R_wc)
        sol.B_wc = float(breakdown.B_wc)
        sol.Op_cost = float(breakdown.Op_cost)
        
        # Final Objective Calculation
        sol.objective_value = (breakdown.objective_total if self.config.use_robust 
                              else breakdown.objective_nominal)
        
        sol.solve_time = time.time() - start_time
        return sol

    # -------------------- Marginal Gain Logic --------------------
    
    def _get_marginal_gain(self, i: int, j: int, t: int, 
                           current_state: Dict, K: int, rho: float, 
                           A_prev: Dict, active_APs: List[int]) -> Tuple[float, float, float, float]:
        """
        Calculates: Objective(current + {j}) - Objective(current)
        Returns: (Total_Gain, Delta_Nominal, Delta_Robust, Cost)
        """
        H_i = self.config.hop_budgets[i]
        
        # --- 1. Nominal Benefit Gain ---
        # Only checking relevant APs to speed this up
        delta_B_nom = 0.0
        
        # Normalization factor (same as in evaluators)
        # We calculate this once roughly or pass it in. 
        # For speed in greedy, we can ignore the denominator scaling IF it's constant, 
        # but to mix with Cost (Op), we need the scale to be roughly correct.
        # Let's approximate scale = 1.0 for ranking, or calculate exactly if needed.
        # To be strictly correct with Lambda_add, we ideally need the H_sum*N_sum denominator.
        # (Assuming H_sum*N_sum is calculated outside or cached).
        # For this snippet, assuming unnormalized raw gain for ranking vs cost.
        
        for p in active_APs:
            req = self.data.counts.get((i, p, t), 0)
            if req <= 0: continue
            
            # Current best hop
            curr_h = self._min_hop_helper(i, p, current_state, H_i)
            
            # New hop if we add j
            dist_j = self.config.hop_distances.get((j, p), 999)
            new_h = min(curr_h, dist_j)
            
            # Gain = Improvement in "closeness" (Benefit = H - hop)
            if new_h < curr_h and new_h <= H_i:
                w = self.data.weights_nominal.get((i, p, t), 1.0)
                gain_raw = (max(0, H_i - new_h) - max(0, H_i - curr_h))
                delta_B_nom += w * req * gain_raw

        # --- 2. Robust Benefit Gain ---
        delta_B_wc = 0.0
        if rho > 0:
            for p in active_APs:
                req = self.data.counts.get((i, p, t), 0)
                if req <= 0: continue
                
                # Calculate current (K+1)-th best hop
                # We do this by getting all hops, sorting, picking index K.
                hops = []
                # Collect hops from EXISTING placements
                for srv in range(self.config.num_servers):
                    if current_state.get((i, srv), 0) == 1:
                        d = self.config.hop_distances.get((srv, p), 999)
                        if d <= H_i:
                            hops.append(d)
                hops.sort()
                
                # Current Robust Hop
                if len(hops) <= K:
                    curr_wc_h = H_i # Effectively infinite/zero benefit
                else:
                    curr_wc_h = hops[K]
                
                # New Robust Hop (simulate adding j)
                dist_j = self.config.hop_distances.get((j, p), 999)
                
                # Insert dist_j into sorted list
                new_hops = sorted(hops + [dist_j])
                if len(new_hops) <= K:
                    new_wc_h = H_i
                else:
                    new_wc_h = new_hops[K] # The K-th index is the (K+1)-th item
                
                if new_wc_h < curr_wc_h and new_wc_h <= H_i:
                    w = self.data.weights_nominal.get((i, p, t), 1.0)
                    gain_raw = (max(0, H_i - new_wc_h) - max(0, H_i - curr_wc_h))
                    delta_B_wc += w * req * gain_raw

        # --- 3. Apply Normalization ---
        # (Re-calculating denom here for correctness)
        I_act = [ii for ii in range(self.config.num_datasets) 
                 if self.data.active_datasets.get((ii, t), 0) == 1]
        H_sum = sum(self.config.hop_budgets[ii] for ii in I_act)
        N_sum = sum(self.data.counts.get((ii, p, t), 0) for ii in I_act for p in active_APs)
        denom = max(H_sum * N_sum, 1.0)
        
        delta_B_nom /= denom
        delta_B_wc /= denom

        # --- 4. Cost Calculation ---
        cost = 0.0
        # Migration Cost
        if self.config.lambda_add > 0 and A_prev.get((i, j), 0) == 0:
            cost += self.config.lambda_add * self.config.get_add_cost(i, j, t)
        
        # Stability Penalty (if used)
        if getattr(self.config, "eta_stability", 0.0) > 0 and A_prev.get((i, j), 0) == 0:
            cost += self.config.eta_stability

        # --- 5. Total Marginal Gain ---
        # Note: We ignore Delta_R (Dedup) in greedy for simplicity, 
        # or add a small heuristic constant if preferred.
        total_gain = (1.0 - rho) * delta_B_nom + rho * delta_B_wc - cost
        
        return total_gain, delta_B_nom, delta_B_wc, cost

    # -------------------- Helpers --------------------
    
    def _min_hop_helper(self, i, p, state, H_i):
        """Quick helper to find current best hop"""
        best = 999
        for srv in range(self.config.num_servers):
            if state.get((i, srv), 0) == 1:
                d = self.config.hop_distances.get((srv, p), 999)
                if d < best:
                    best = d
        return min(best, H_i)

    # -------------------- Evaluators (Standard) --------------------
    # Copying your standard evaluators to ensure this class is self-contained
    
    def _compute_nominal_objectives(self, sol, t, P_t, A_prev):
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        
        # R_nom
        denom = sum(self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                    for i in I for j in J if self.data.active_datasets.get((i, t), 0) == 1)
        numer = sum(self.config.dataset_sizes[i] * sol.removes.get((i, j), 0) 
                    for i in I for j in J)
        R_nom = numer / max(denom, 1.0)

        # B_nom
        H_sum = sum(self.config.hop_budgets[i] for i in I 
                    if self.data.active_datasets.get((i, t), 0) == 1)
        N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I for p in P_t)
        denom_B = max(H_sum * N_sum, 1.0)
        
        bsum = 0.0
        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1: continue
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                req = self.data.counts.get((i, p, t), 0)
                if req > 0:
                    hop = self._min_hop_helper(i, p, sol.states, H_i)
                    bsum += req * max(H_i - hop, 0) * self.data.weights_nominal.get((i,p,t),1.0)
        B_nom = bsum / denom_B
        
        # Op
        Op = self.config.lambda_add * sum(self.config.get_add_cost(i, j, t) * sol.adds.get((i, j), 0)
                                          for i in I for j in J)
        return R_nom, B_nom, Op

    def _compute_B_wc_from_state(self, state, t, P_t):
        I = range(self.config.num_datasets)
        K = self.config.K_failures
        
        H_sum = sum(self.config.hop_budgets[i] for i in I 
                    if self.data.active_datasets.get((i, t), 0) == 1)
        N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I for p in P_t)
        denom_B = max(H_sum * N_sum, 1.0)
        
        bsum = 0.0
        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1: continue
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                req = self.data.counts.get((i, p, t), 0)
                if req > 0:
                    # K-knockout logic
                    hops = [self.config.hop_distances.get((j, p), 999) 
                            for j in range(self.config.num_servers) 
                            if state.get((i,j),0)==1]
                    hops = [h for h in hops if h <= H_i]
                    hops.sort()
                    
                    wc_hop = hops[K] if len(hops) > K else H_i
                    bsum += req * max(H_i - wc_hop, 0) * self.data.weights_nominal.get((i,p,t),1.0)
        return bsum / denom_B

    def _compute_R_wc_from_state(self, state, t, A_prev):
        # Simple approximation: 0 or reuse R_nom / (K+1) if you want a specific metric
        return 0.0
