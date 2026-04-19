from solver_base import BaseSolver,GlobalSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from milp_solver import MILPSolver
from offline_milp_solver import OfflineMILPSolver
from beam_search_solver import BeamSearchSolver
from lyapunov_solver import LyapunovSolver
from greedy_custom_solver import GreedyCustomSolver
from rein_adaptive_solver import REINAdaptiveSolver
from mcts_solver import MCTSSolver
from q_learning_solver import QLearningSolver
from kmeans_dataset_solver import KMeansDatasetSolver
from adaptive_ensemble_solver import AdaptiveEnsembleSolver
from typing import Dict, Tuple, List, Optional, Union
from results_visualizer import ResultsVisualizer
import copy

class SolverFactory:
    """Factory for creating solvers based on configuration"""
    
    @staticmethod
    def create_solver(config: uEDDEConfig, data_gen: DataGenerator) -> BaseSolver:
        """Create solver based on config.solver_type"""
        solver_type = config.solver_type.lower()
        
        if solver_type == "milp":
            return MILPSolver(config, data_gen)
        elif solver_type == "offline_milp":
            return OfflineMILPSolver(config, data_gen)
        elif solver_type == "beam_search":
            return BeamSearchSolver(config, data_gen)
        elif solver_type in ["lyapunovsolver", "lyapunov"]:
            return LyapunovSolver(config, data_gen)
        elif solver_type == "rein":
            return REINAdaptiveSolver(config, data_gen)
        elif solver_type == "greedy":
            return GreedyCustomSolver(config,data_gen)
        elif solver_type == "mcts":
            return MCTSSolver(config, data_gen)
        elif solver_type == "qlearning":
            return QLearningSolver(config, data_gen)
        elif solver_type in ["kmeans_dataset", "kmeans++", "kmeans"]:
            return KMeansDatasetSolver(config, data_gen)
        elif solver_type in ["adaptive_ensemble", "ensemble", "meta"]:
            return AdaptiveEnsembleSolver(config, data_gen)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}. "
                           f"Options: 'milp', 'offline_milp', 'beam_search', 'hindsight_online_milp', 'greedy', 'lyapunov' (or 'lyapunovsolver'), 'rein', 'mcts', 'qlearning', 'gnn_ppo', 'kmeans_dataset', 'adaptive_ensemble'")
    @staticmethod
    def _run_single_configuration(config: uEDDEConfig, data_gen: DataGenerator,
                                  show_visuals: bool = True,
                                  k_context: Optional[Tuple[int, List[Tuple[int, float]]]] = None) -> GlobalSolution:
        """Internal helper to solve once and optionally visualize"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*20 + "ONLINE uEDDE SOLVER" + " "*39 + "║")
        print("╚" + "="*78 + "╝")
        
        config.print_summary()
        data_gen.print_summary()
        
        solver = SolverFactory.create_solver(config, data_gen)
        solution = solver.solve_all()
        
        if k_context is not None:
            k_value, storage = k_context
            storage.append((int(k_value), solution.total_objective))
            sweep_payload = storage
        else:
            sweep_payload = None
        
        if show_visuals and config.plot_per_slot:
            solver.print_solution()
        
        if show_visuals and config.plot_summary:
            viz = ResultsVisualizer(solution, data_gen)
            viz.print_statistics_table()
            viz.plot_all(k_sweep_results=sweep_payload)
        
        return solution

    @staticmethod
    def solve_and_visualize(config: uEDDEConfig, data_gen: DataGenerator,
                            k_list: Optional[List[int]] = None) -> Union[GlobalSolution, Dict[int, GlobalSolution]]:
        """Convenience method: solve, visualize, and optionally sweep over K values"""
        if k_list is None:
            return SolverFactory._run_single_configuration(config, data_gen)
        
        k_values = list(k_list)
        if len(k_values) == 0:
            raise ValueError("k_list must contain at least one value")
        
        normalized_k = []
        for raw_k in k_values:
            if raw_k is None:
                raise ValueError("k_list cannot contain None values")
            k_int = int(raw_k)
            if k_int < 0:
                raise ValueError("k_list must contain non-negative integers")
            normalized_k.append(k_int)
        
        k_results: List[Tuple[int, float]] = []
        solutions: Dict[int, GlobalSolution] = {}
        total_runs = len(normalized_k)
        for idx, k_val in enumerate(normalized_k, start=1):
            print(f"\n>>> Sweep run {idx}/{total_runs} with K={k_val}")
            config_copy = copy.deepcopy(config)
            config_copy.K_failures = k_val
            show_visuals = (idx == total_runs)
            solution = SolverFactory._run_single_configuration(
                config_copy,
                data_gen,
                show_visuals=show_visuals,
                k_context=(k_val, k_results)
            )
            solutions[k_val] = solution
        
        if total_runs == 1:
            return solutions[normalized_k[0]]
        return solutions

