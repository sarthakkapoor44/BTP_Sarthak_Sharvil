"""
Centralized Objective Calculator for uEDDE solvers.
Provides a single source of truth for calculating objective values, preventing
inconsistencies across different solver implementations.
"""

from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class ObjectiveBreakdown:
    """Structured result containing all objective components"""
    R_nominal: float = 0.0  # Dedup ratio (nominal)
    B_nominal: float = 0.0  # Benefit (nominal)
    R_wc: float = 0.0       # Dedup ratio (worst-case)
    B_wc: float = 0.0       # Benefit (worst-case)
    Op_cost: float = 0.0    # Operation/migration cost
    
    # Derived values
    objective_nominal: float = 0.0  # R_nom + B_nom - Op
    objective_wc: float = 0.0       # R_wc + B_wc
    objective_total: float = 0.0    # (1-rho)*(R_nom+B_nom) + rho*(R_wc+B_wc) - Op
    
    def __str__(self) -> str:
        return (
            f"R_nom={self.R_nominal:.4f} | B_nom={self.B_nominal:.4f} | "
            f"R_wc={self.R_wc:.4f} | B_wc={self.B_wc:.4f} | "
            f"Op={self.Op_cost:.4f} | Obj={self.objective_total:.4f}"
        )


class ObjectiveCalculator:
    """
    Unified objective calculator for all solvers.
    Given a configuration, data, current and previous placements, computes all objective metrics.
    """
    
    def __init__(self, config, data_gen):
        """
        Args:
            config: uEDDEConfig instance
            data_gen: DataGenerator instance
        """
        self.config = config
        self.data = data_gen
    
    def calculate(self, 
                  t: int,
                  state_current: Dict[Tuple[int, int], int],
                  state_prev: Dict[Tuple[int, int], int],
                  rho: Optional[float] = None,
                  lambda_add: Optional[float] = None) -> ObjectiveBreakdown:
        """
        Calculate complete objective breakdown for a single time slot.
        
        Args:
            t: Time slot
            state_current: Current placement {(i,j): 1/0, ...}
            state_prev: Previous placement {(i,j): 1/0, ...}
            rho: Override blend parameter (uses config.rho if None)
            lambda_add: Override cost weight (uses config.lambda_add if None)
        
        Returns:
            ObjectiveBreakdown with all components calculated
        """
        rho = rho if rho is not None else self.config.rho
        lambda_add = lambda_add if lambda_add is not None else self.config.lambda_add
        
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])
        
        # ---- R_NOMINAL (Dedup Ratio) ----
        R_nom = self._calculate_R_nominal(I, J, t, state_current, state_prev)
        
        # ---- B_NOMINAL (Benefit) ----
        B_nom = self._calculate_B_nominal(I, J, P_t, t, state_current)
        
        # ---- OPERATION COST ----
        Op = self._calculate_operation_cost(I, J, t, state_current, state_prev, lambda_add)
        
        # ---- WORST-CASE (if robust) ----
        if self.config.use_robust:
            R_wc = self._calculate_R_wc(I, J, t, state_current)
            B_wc = self._calculate_B_wc(I, J, P_t, t, state_current)
        else:
            R_wc, B_wc = 0.0, 0.0
        
        # ---- ASSEMBLE RESULT ----
        result = ObjectiveBreakdown(
            R_nominal=R_nom,
            B_nominal=B_nom,
            R_wc=R_wc,
            B_wc=B_wc,
            Op_cost=Op,
        )
        
        # Compute derived objectives
        result.objective_nominal = R_nom + B_nom - Op
        result.objective_wc = R_wc + B_wc
        result.objective_total = (1.0 - rho) * (R_nom + B_nom) + rho * result.objective_wc - Op
        
        return result
    
    # ========== NOMINAL COMPONENTS ==========
    
    def _calculate_R_nominal(self, I, J, t, state_current, state_prev) -> float:
        """
        Calculate R_nominal: Dedup ratio = removed_size / previous_size
        over active datasets only.
        """
        I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
        
        # Denominator: total size in previous state (active datasets only)
        denom = sum(
            self.config.dataset_sizes[i] * state_prev.get((i, j), 0)
            for i in I_act for j in J
        )
        
        if denom <= 0:
            # print("⚠️  Warning: R_nominal denominator is zero or negative. Returning 0.0.")
            return 0.0
        
        # Numerator: total size removed
        numer = sum(
            self.config.dataset_sizes[i]
            for i in I_act for j in J
            if state_prev.get((i, j), 0) == 1 and state_current.get((i, j), 0) == 0
        )
        
        return numer / denom
    
    def _calculate_B_nominal(self, I, J, P_t, t, state_current) -> float:
        """
        Calculate B_nominal: Benefit from proximity.
        B = sum over (i,p) of w_bar(i,p,t) * n(i,p,t) * max(0, H_i - min_hop)
        Normalized by: (sum H_i over active i) * (sum n(i,p,t) over all i,p)
        """
        I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
        
        # Normalization factors
        H_sum = sum(self.config.hop_budgets[i] for i in I_act)
        N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)
        
        if H_sum <= 0 or N_sum <= 0:
            return 0.0
        
        # Numerator: sum of (weight * count * benefit)
        total = 0.0
        for i in I_act:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                req = self.data.counts.get((i, p, t), 0)
                if req <= 0:
                    continue
                
                # Find minimum hop from any server storing i to AP p
                min_hop = float('inf')
                for j in J:
                    if state_current.get((i, j), 0) == 1:
                        hop_dist = self.config.hop_distances.get((j, p), float('inf'))
                        if hop_dist < min_hop:
                            min_hop = hop_dist
                
                # Clamp to hop budget
                if min_hop <= H_i:
                    benefit = H_i - min_hop
                    weight = self.data.weights_nominal.get((i, p, t), 1.0)
                    total += weight * req * benefit
        
        return total / (H_sum * N_sum)
    
    def _calculate_operation_cost(self, I, J, t, state_current, 
                                  state_prev, lambda_add) -> float:
        """
        Calculate operation cost: sum of costs for all adds (prev=0 -> curr=1).
        """
        if lambda_add <= 0:
            return 0.0
        
        cost = 0.0
        for i in I:
            for j in J:
                if state_prev.get((i, j), 0) == 0 and state_current.get((i, j), 0) == 1:
                    add_cost = self.config.get_add_cost(i, j, t)
                    cost += lambda_add * add_cost
        
        return cost
    
    # ========== WORST-CASE COMPONENTS ==========
    
    def _calculate_R_wc(self, I, J, t, state_current) -> float:
        """
        Calculate R_wc: Worst-case dedup ratio under K failures.
        Simplified: use nominal denominator (previous placement not changed by adversary in WC).
        """
        # For WC, we typically assume the adversary knocks out K servers
        # Resulting in removals from those servers.
        # A detailed WC calculation would enumerate scenarios; here we use a placeholder.
        # More sophisticated: iterate scenarios and compute for each.
        
        # SIMPLIFIED: return 0.0 (no removals in WC scenario by assumption)
        # TODO: If needed, enumerate failure scenarios and compute actual R_wc
        return 0.0
    
    def _calculate_B_wc(self, I, J, P_t, t, state_current) -> float:
        """
        Calculate B_wc: Worst-case benefit when up to K servers fail.
        For each (i,p), benefit is determined by the (K+1)-th closest server.
        """
        I_act = [i for i in I if self.data.active_datasets.get((i, t), 0) == 1]
        
        # Normalization factors (same as nominal)
        H_sum = sum(self.config.hop_budgets[i] for i in I_act)
        N_sum = sum(self.data.counts.get((i, p, t), 0) for i in I_act for p in P_t)
        
        if H_sum <= 0 or N_sum <= 0:
            return 0.0
        
        K = self.config.K_failures
        total = 0.0
        
        for i in I_act:
            H_i = self.config.hop_budgets[i]
            for p in P_t:
                req = self.data.counts.get((i, p, t), 0)
                if req <= 0:
                    continue
                
                # Collect all hops from servers storing i to AP p
                hops = []
                for j in J:
                    if state_current.get((i, j), 0) == 1:
                        hop_dist = self.config.hop_distances.get((j, p), float('inf'))
                        if hop_dist <= H_i:
                            hops.append(hop_dist)
                
                hops.sort()
                
                # WC hop is the (K+1)-th smallest (survives K failures)
                if len(hops) <= K:
                    wc_hop = H_i  # All servers can fail; no benefit
                else:
                    wc_hop = hops[K]  # K-th index = (K+1)-th value
                
                if wc_hop <= H_i:
                    benefit = H_i - wc_hop
                    weight = self.data.weights_nominal.get((i, p, t), 1.0)
                    total += weight * req * benefit
        
        return total / (H_sum * N_sum)
    
    # ========== HELPER METHODS ==========
    
    def print_breakdown(self, breakdown: ObjectiveBreakdown, prefix: str = ""):
        """Pretty-print an objective breakdown"""
        print(f"{prefix}R_nom={breakdown.R_nominal:.6f}  B_nom={breakdown.B_nominal:.6f}  "
              f"R_wc={breakdown.R_wc:.6f}  B_wc={breakdown.B_wc:.6f}  "
              f"Op={breakdown.Op_cost:.6f}")
        print(f"{prefix}→ Total Objective: {breakdown.objective_total:.6f}")


# ============================================================================
# USAGE EXAMPLE (in any solver):
# ============================================================================
#
# In solve_slot() method:
#
#     calculator = ObjectiveCalculator(self.config, self.data)
#     breakdown = calculator.calculate(t, state_current, state_prev)
#
#     # Access components
#     sol.R_nominal = breakdown.R_nominal
#     sol.B_nominal = breakdown.B_nominal
#     sol.R_wc = breakdown.R_wc
#     sol.B_wc = breakdown.B_wc
#     sol.Op_cost = breakdown.Op_cost
#     sol.objective_value = breakdown.objective_total
#
# This ensures all solvers use identical objective calculation logic.
#
