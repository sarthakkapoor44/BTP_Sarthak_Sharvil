from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from config import uEDDEConfig
from data_generator import DataGenerator

from abc import ABC, abstractmethod

@dataclass
class SlotSolution:
    """Solution for a single time slot"""
    time_slot: int
    adds: Dict[Tuple[int, int], int] = field(default_factory=dict)  # a_{ij,t}
    removes: Dict[Tuple[int, int], int] = field(default_factory=dict)  # r_{ij,t}
    states: Dict[Tuple[int, int], int] = field(default_factory=dict)  # x_{ij,t}
    
    # Worst-case scenario info
    failures: Dict[int, int] = field(default_factory=dict)  # f_{j,t}
    survivors: Dict[Tuple[int, int], float] = field(default_factory=dict)  # y_{ij,t}
    recourse_removes: Dict[Tuple[int, int], int] = field(default_factory=dict)  # q_{ij,t}
    post_recourse: Dict[Tuple[int, int], int] = field(default_factory=dict)  # z_{ij,t}
    
    # Ring variables
    ring_nominal: Dict[Tuple[int, int, int], int] = field(default_factory=dict)  # w^nom_{ipd}
    ring_wc: Dict[Tuple[int, int, int], int] = field(default_factory=dict)  # w^wc_{ipd}
    
    # Benefits
    benefit_nominal_per_ip: Dict[Tuple[int, int], float] = field(default_factory=dict)  # B^nom_{ip,t}
    benefit_wc_per_ip: Dict[Tuple[int, int], float] = field(default_factory=dict)  # B^wc_{ip,t}
    
    # Objectives
    R_nominal: float = 0.0
    R_wc: float = 0.0
    B_nominal: float = 0.0
    B_wc: float = 0.0
    Op_cost: float = 0.0
    objective_value: float = 0.0
    
    # Metadata
    solve_time: float = 0.0
    status: str = "unknown"
    ccg_iterations: int = 0

@dataclass
class GlobalSolution:
    """Complete solution across all time slots"""
    config: uEDDEConfig
    slot_solutions: List[SlotSolution] = field(default_factory=list)
    total_objective: float = 0.0
    total_solve_time: float = 0.0
    
    def add_slot_solution(self, sol: SlotSolution):
        self.slot_solutions.append(sol)
        self.total_objective += sol.objective_value
        self.total_solve_time += sol.solve_time


class BaseSolver(ABC):
    """Abstract base class for all solvers"""
    
    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        self.config = config
        self.data = data_gen
        self.solution = GlobalSolution(config=config)
    
    @abstractmethod
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        """Solve for a single time slot given previous state"""
        pass
    
    def solve_all(self) -> GlobalSolution:
        """Solve across all time slots (online)"""
        verbose = getattr(self.config, "verbose", True)
        if verbose:
            print("\n" + "="*80)
            print(f"SOLVING ONLINE uEDDE [{self.__class__.__name__}]")
            print("="*80)
        
        A_prev = self.data.initial_state.copy()
        log_every = 1 if self.config.T <= 50 else max(1, self.config.T // 20)
        
        for t in range(1, self.config.T + 1):
            should_log = verbose and (t == 1 or t == self.config.T or t % log_every == 0)
            if should_log:
                print(f"\n{'─'*80}")
                print(f"Time Slot {t}/{self.config.T}")
                print(f"{'─'*80}")
            
            slot_sol = self.solve_slot(t, A_prev)
            self.solution.add_slot_solution(slot_sol)
            
            # Update state for next slot
            A_prev = slot_sol.states.copy()
            
            if should_log:
                print(f"✓ Slot {t} complete: Obj={slot_sol.objective_value:.4f}, "
                      f"Adds={len(slot_sol.adds)}, Removes={len(slot_sol.removes)}, "
                      f"Time={slot_sol.solve_time:.2f}s")
        
        if verbose:
            print("\n" + "="*80)
            print(f"TOTAL OBJECTIVE: {self.solution.total_objective:.4f}")
            print(f"TOTAL TIME: {self.solution.total_solve_time:.2f}s")
            print("="*80)
        
        return self.solution
    
    def print_solution(self):
        """Print detailed solution report"""
        from datetime import datetime
        
        print("\n" + "="*80)
        print("DETAILED SOLUTION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Solver: {self.__class__.__name__}")
        print(f"Config: T={self.config.T}, I={self.config.num_datasets}, "
              f"J={self.config.num_servers}, K={self.config.K_failures}, "
              f"Γ={self.config.Gamma_budget}, ρ={self.config.rho}")
        print("="*80)
        
        for slot_sol in self.solution.slot_solutions:
            self._print_slot_solution(slot_sol)
    
    def _print_slot_solution(self, sol: SlotSolution):
        """Print solution for one slot"""
        t = sol.time_slot
        print(f"\n{'═'*80}")
        print(f"TIME SLOT t={t}")
        print(f"{'═'*80}")
        
        # Requests
        print(f"\n📊 Request Distribution:")
        for p in self.data.attachment_points.get(t, []):
            total_requests = sum(self.data.counts.get((i, p, t), 0) 
                                for i in range(self.config.num_datasets))
            if total_requests > 0:
                print(f"  AP {p}: {total_requests} requests")
                for i in range(self.config.num_datasets):
                    count = self.data.counts.get((i, p, t), 0)
                    if count > 0:
                        weight = self.data.weights_nominal.get((i, p, t), 1.0)
                        print(f"    Dataset {i}: {count} reqs (weight={weight:.3f})")
        
        # First-stage decisions
        print(f"\n📥 First-Stage Decisions:")
        if sol.adds:
            print(f"  Adds ({len(sol.adds)}):")
            for (i, j), val in sorted(sol.adds.items()):
                if val > 0:
                    cost = self.config.get_add_cost(i, j, t)
                    print(f"    Dataset {i} → Server {j} (cost={cost:.2f})")
        else:
            print(f"  Adds: none")
        
        if sol.removes:
            print(f"  Removes ({len(sol.removes)}):")
            for (i, j), val in sorted(sol.removes.items()):
                if val > 0:
                    print(f"    Dataset {i} ← Server {j}")
        else:
            print(f"  Removes: none")
        
        # Post-first-stage placement
        print(f"\n📍 Post-First-Stage Placement:")
        for i in range(self.config.num_datasets):
            servers = [j for (ii, j), val in sol.states.items() if ii == i and val > 0]
            if servers:
                print(f"  Dataset {i}: servers {sorted(servers)}")
            else:
                print(f"  Dataset {i}: not placed")
        
        # Worst-case scenario
        if self.config.use_robust and sol.failures:
            print(f"\n⚠️  Worst-Case Scenario:")
            failed = [j for j, val in sol.failures.items() if val > 0]
            print(f"  Failed servers: {failed if failed else 'none'}")
            
            if sol.recourse_removes:
                print(f"  Recourse removes:")
                for (i, j), val in sorted(sol.recourse_removes.items()):
                    if val > 0:
                        print(f"    Dataset {i} ← Server {j}")
        
        # Objective components
        print(f"\n🎯 Objective Components:")
        print(f"  R_nominal (dedup ratio): {sol.R_nominal:.4f}")
        print(f"  B_nominal (benefit): {sol.B_nominal:.4f}")
        if self.config.use_robust:
            print(f"  R_wc (worst-case dedup): {sol.R_wc:.4f}")
            print(f"  B_wc (worst-case benefit): {sol.B_wc:.4f}")
        print(f"  Op_cost (migration): {sol.Op_cost:.4f}")
        print(f"  ▶ Total Objective: {sol.objective_value:.4f}")
        
        print(f"\n⏱️  Solve time: {sol.solve_time:.2f}s, Status: {sol.status}")

