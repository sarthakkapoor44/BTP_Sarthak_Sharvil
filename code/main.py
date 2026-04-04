#!/usr/bin/env python3
"""
Main entry point for the uEDDE Online Data Placement Solver.
Defines parameters, configures solvers, and executes experiments.

DATASET MODES:
==============

1. SYNTHETIC MODE:
   exp.DATASET_SOURCE = "synthetic"
   Generates random data. Set T, NUM_DATASETS, NUM_SERVERS manually.

2. CSV MODE:
   exp.DATASET_SOURCE = "csv"
   exp.CSV_FILE = "data.csv"  (columns: timestamp, server, dataset)
   
   Automatically derives:
     - T = max(timestamp) + 1
     - NUM_SERVERS = number of unique servers
     - NUM_DATASETS = number of unique datasets
   
   CSV Format Example:
     timestamp,server,dataset
     0,7,4
     0,8,4
     0,5,4
     1,7,5
     1,3,2

Example Usage:
  exp = ExperimentConfig()
  exp.DATASET_SOURCE = "csv"
  exp.CSV_FILE = "200netflix.csv"
  
  config = build_config(exp)  # Derives dimensions from CSV
  data = build_data(config)   # Loads data
"""

import copy
import numpy as np
from typing import Dict, List, Optional

# ============================================================================
# IMPORTS FROM MODULAR FILES
# ============================================================================
from config import uEDDEConfig
from graphGen import generate_random_connected_adjacency, visualize_adjacency
from data_generator import DataGenerator
from solver_base import BaseSolver, GlobalSolution
from solver_factory import SolverFactory
from results_visualizer import ResultsVisualizer
from objective_calculator import ObjectiveCalculator, ObjectiveBreakdown
from solver_comparison import compare_two_solvers, run_solver
from sensitivity_analysis import run_sensitivity_analysis
from migration_analysis import extract_migration_df, plot_migration_costs_separate
from solver_benchmark import run_solvers_and_collect, plot_comparison, run_solver_types_and_collect, run_k_sweep_parallel
from online_solver_visualization import visualize_solvers

# ============================================================================
# PARAMETER CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Central hub for experiment parameters"""
    
    # -------- TIME & PROBLEM SIZE --------
    T: int = 5  # Number of time slots
    NUM_DATASETS: int = 3
    NUM_SERVERS: int = 5
    
    # -------- ROBUSTNESS --------
    K_FAILURES: int = 1  # Maximum failures per slot
    GAMMA_BUDGET: float = 2.0  # Bertsimas-Sim uncertainty budget
    RHO: float = 0.5  # Blend: 0=nominal only, 1=worst-case only
    USE_ROBUST: bool = True
    USE_CCG: bool = True
    
    # -------- OPTIMIZATION --------
    LAMBDA_ADD: float = 1.0  # Migration cost weight
    ETA_STABILITY: float = 0.0  # Stability penalty
    SOLVER_TYPE: str = "milp"  # Options: "milp", "offline_milp", "custom", "greedy", "lyapunov", "mcts", "qlearning"
    SOLVER_TIME_LIMIT: int = 300  # Seconds
    SOLVER_GAP: float = 0.01  # MIP gap tolerance
    
    # -------- NETWORK --------
    EDGE_PROBABILITY: float = 0.5  # For random topology
    HOP_BUDGET_VALUE: int = 1  # Uniform hop budget applied to all datasets by default
    HOP_BUDGETS: Dict[int, int] = None  # Optional explicit per-dataset override
    
    # -------- CAPACITY --------
    SERVER_CAPACITY: float = 10000.0  # Per-server capacity
    DATASET_SIZE: float = 10.0  # Per-dataset size
    
    # -------- DATA --------
    DATASET_SOURCE: str = "synthetic"  # Options: "synthetic", "csv"
    CSV_FILE: Optional[str] = None  # Path to CSV file (required if DATASET_SOURCE="csv")
    
    # -------- VISUALIZATION --------
    PLOT_PER_SLOT: bool = True
    PLOT_SUMMARY: bool = True
    SAVE_OUTPUTS: bool = True
    OUTPUT_DIR: str = "results"

    # -------- EXECUTION --------
    PARALLEL: bool = True
    N_JOBS: int = 0  # 0 = auto(cpu_count), 1 = force sequential
    VERBOSE: bool = True
    
    def resolve_hop_budgets(self) -> Dict[int, int]:
        """Return hop budgets from one central control point.

        Priority:
        1) Explicit HOP_BUDGETS dict (if provided)
        2) Uniform HOP_BUDGET_VALUE for all datasets
        """
        if self.HOP_BUDGETS is not None:
            # Normalize to full dataset index range in case caller provides partial map.
            return {
                i: int(self.HOP_BUDGETS.get(i, self.HOP_BUDGET_VALUE))
                for i in range(self.NUM_DATASETS)
            }
        return {i: int(self.HOP_BUDGET_VALUE) for i in range(self.NUM_DATASETS)}


# ============================================================================
# BUILDER FUNCTIONS
# ============================================================================

def build_config(exp_cfg: ExperimentConfig) -> uEDDEConfig:
    """
    Build a uEDDEConfig from experiment parameters.
    
    For CSV mode: Automatically derives T, NUM_DATASETS, NUM_SERVERS from CSV file.
    """
    config = uEDDEConfig()
    
    # -------- FOR CSV MODE: DERIVE DIMENSIONS FROM FILE --------
    if exp_cfg.DATASET_SOURCE == "csv":
        if not exp_cfg.CSV_FILE:
            raise ValueError("CSV_FILE must be provided when DATASET_SOURCE='csv'")
        
        # Peek at CSV to get dimensions
        import pandas as pd
        print(f"Reading CSV to derive dimensions: {exp_cfg.CSV_FILE}")
        try:
            df = pd.read_csv(exp_cfg.CSV_FILE)
            
            # Validate required columns
            if not all(col in df.columns for col in ['timestamp', 'server', 'dataset']):
                raise ValueError(f"CSV must have columns: timestamp, server, dataset")
            
            # Derive dimensions
            actual_T = int(df['timestamp'].max()) + 1  # T = number of time slots
            actual_servers = len(df['server'].unique())  # NUM_SERVERS
            actual_datasets = len(df['dataset'].unique())  # NUM_DATASETS
            
            print(f"  ✓ Derived from CSV:")
            print(f"    T (time slots) = {actual_T} (from timestamp 0 to {actual_T-1})")
            print(f"    Servers = {actual_servers} (unique server IDs: {sorted(df['server'].unique().tolist())})")
            print(f"    Datasets = {actual_datasets} (unique dataset IDs: {sorted(df['dataset'].unique().tolist())})")
            
            # Update experiment config with derived values
            exp_cfg.T = actual_T
            exp_cfg.NUM_SERVERS = actual_servers
            exp_cfg.NUM_DATASETS = actual_datasets
            
        except Exception as e:
            print(f"✗ Error reading CSV: {e}")
            raise
    
    # -------- GENERATE NETWORK TOPOLOGY --------
    adjacency = generate_random_connected_adjacency(
        num_servers=exp_cfg.NUM_SERVERS,
        edge_probability=exp_cfg.EDGE_PROBABILITY,
        seed=42
    )
    
    # -------- RESOLVE HOP BUDGETS FROM CENTRAL CONTROL --------
    hop_budgets = exp_cfg.resolve_hop_budgets()

    # -------- OVERRIDE CONFIG WITH PARAMETERS --------
    config.override(
        T=exp_cfg.T,
        num_datasets=exp_cfg.NUM_DATASETS,
        num_servers=exp_cfg.NUM_SERVERS,
        rho=exp_cfg.RHO,
        K_failures=exp_cfg.K_FAILURES,
        Gamma_budget=exp_cfg.GAMMA_BUDGET,
        lambda_add=exp_cfg.LAMBDA_ADD,
        eta_stability=exp_cfg.ETA_STABILITY,
        use_robust=exp_cfg.USE_ROBUST,
        use_ccg=exp_cfg.USE_CCG,
        solver_type=exp_cfg.SOLVER_TYPE,
        solver_time_limit=exp_cfg.SOLVER_TIME_LIMIT,
        solver_gap=exp_cfg.SOLVER_GAP,
        plot_per_slot=exp_cfg.PLOT_PER_SLOT,
        plot_summary=exp_cfg.PLOT_SUMMARY,
        save_plots=exp_cfg.SAVE_OUTPUTS,
        plot_output_dir=exp_cfg.OUTPUT_DIR,
        verbose=exp_cfg.VERBOSE,
        hop_budgets=hop_budgets,
        dataset_sizes={i: exp_cfg.DATASET_SIZE for i in range(exp_cfg.NUM_DATASETS)},
        server_capacities={j: exp_cfg.SERVER_CAPACITY for j in range(exp_cfg.NUM_SERVERS)},
        adjacency=adjacency,
        dataset_source=exp_cfg.DATASET_SOURCE,
        custom_dataset_path=exp_cfg.CSV_FILE,  # Store CSV file path for later loading
    )

    if exp_cfg.VERBOSE:
        print(f"✓ Hop budget control: uniform={exp_cfg.HOP_BUDGET_VALUE}, resolved={hop_budgets}")
    
    if exp_cfg.VERBOSE:
        config.print_summary()
    
    return config


def build_data(config: uEDDEConfig, seed: int = 42) -> DataGenerator:
    """
    Build and populate a DataGenerator.
    
    Modes:
      - "synthetic": Generates random data
      - "csv": Loads from CSV file (CSV_FILE path stored in config.custom_dataset_path)
    """
    data = DataGenerator(config, seed=seed)
    
    if config.dataset_source == "synthetic":
        # Generate synthetic random data
        data.generate_all("synthetic", activity=1.0)
        if config.verbose:
            print("✓ Generated synthetic data")
        
    elif config.dataset_source == "csv":
        data.generate_all("data", activity=1)
        if config.verbose:
            print("✓ Loaded data from CSV")
        
    else:
        raise ValueError(f"Unknown dataset_source: {config.dataset_source}. Use 'synthetic' or 'csv'")
    
    if hasattr(config, 'verbose') and config.verbose:
        data.print_summary()
    
    return data


def validate_csv_format(csv_path: str) -> bool:
    """Validate that a CSV file has the required format for data loading"""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, nrows=5)
        required_cols = ['timestamp', 'server', 'dataset']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"✗ Missing required columns: {missing}")
            print(f"  Found columns: {list(df.columns)}")
            print(f"  Expected format: timestamp,server,dataset")
            return False
        
        print(f"✓ CSV format valid: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        
        # Show sample
        print(f"  Sample rows:")
        for idx, row in df.head(3).iterrows():
            print(f"    {row['timestamp']},{row['server']},{row['dataset']}")
        
        return True
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False


def demonstrate_objective_calculator(exp_cfg: ExperimentConfig) -> ObjectiveBreakdown:
    """
    Demonstrate the centralized ObjectiveCalculator.
    Shows how all solvers should use this for consistent calculations.
    """
    print("\n" + "="*80)
    print("OBJECTIVE CALCULATOR DEMO")
    print("="*80)
    print("This demonstrates the benefit of using a centralized ObjectiveCalculator")
    print("instead of each solver reimplementing objective calculation logic.\n")
    
    config = build_config(exp_cfg)
    data = build_data(config)
    
    # Create calculator
    calc = ObjectiveCalculator(config, data)
    
    # Example: calculate objectives for slot 1
    t = 1
    state_prev = data.initial_state.copy()
    state_current = state_prev.copy()
    
    # Simulate an add: dataset 0 to server 1
    state_current[(0, 1)] = 1
    
    breakdown = calc.calculate(t, state_current, state_prev)
    
    print(f"Time Slot: {t}")
    print(f"Previous State: {len(state_prev)} placements")
    print(f"Current State: {len(state_current)} placements (added 1 replica)")
    print(f"\n{breakdown}")
    print("\n✓ All solvers should now use ObjectiveCalculator instead of")
    print("  reimplementing this logic independently.\n")
    
    return breakdown


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_single_solver(exp_cfg: ExperimentConfig) -> GlobalSolution:
    """Run a single solver with given configuration"""
    print("\n" + "="*80)
    print("SINGLE SOLVER RUN")
    print("="*80)
    
    config = build_config(exp_cfg)
    data = build_data(config)
    
    solver = SolverFactory.create_solver(config, data)
    solution = solver.solve_all()
    
    if config.plot_per_slot and config.verbose:
        solver.print_solution()
    
    if config.plot_summary:
        viz = ResultsVisualizer(solution, data)
        viz.print_statistics_table(save_to_file=True)
        viz.plot_all()
    
    return solution


def run_solver_pair_comparison(exp_cfg_a: ExperimentConfig, exp_cfg_b: ExperimentConfig) -> Dict:
    """Compare two solvers side-by-side (fair comparison with same data)"""
    print("\n" + "="*80)
    print("SOLVER PAIR COMPARISON")
    print("="*80)
    
    # Build config and data ONCE (to be fair)
    config = build_config(exp_cfg_a)
    data_ref = build_data(config)
    
    # Create two solvers using deep copies of data
    solver_a = SolverFactory.create_solver(config, copy.deepcopy(data_ref))
    
    # For second solver, override solver type if different
    config_b = copy.deepcopy(config)
    config_b.override(solver_type=exp_cfg_b.SOLVER_TYPE)
    solver_b = SolverFactory.create_solver(config_b, copy.deepcopy(data_ref))
    
    summary = compare_two_solvers(solver_a, solver_b, config, data_ref, plot=True)
    return summary


def run_k_failure_sweep(exp_cfg: ExperimentConfig, k_values: List[int]) -> Dict[int, GlobalSolution]:
    """Sweep across failure budgets K"""
    print("\n" + "="*80)
    print(f"K-FAILURE SWEEP: {k_values}")
    print("="*80)
    
    config = build_config(exp_cfg)
    data = build_data(config)

    if exp_cfg.PARALLEL and len(k_values) > 1:
        solutions = run_k_sweep_parallel(
            base_config=config,
            base_data=data,
            k_values=k_values,
            solver_type=exp_cfg.SOLVER_TYPE,
            seed=42,
            n_jobs=exp_cfg.N_JOBS,
        )
        # Keep sweep visualization behavior: visualize using the largest K solution.
        if config.plot_summary and solutions:
            k_sorted = sorted(solutions.keys())
            k_payload = [(k, solutions[k].total_objective) for k in k_sorted]
            viz = ResultsVisualizer(solutions[k_sorted[-1]], data)
            viz.print_statistics_table(save_to_file=True)
            viz.plot_all(k_sweep_results=k_payload)
    else:
        solutions = SolverFactory.solve_and_visualize(config, data, k_list=k_values)
    return solutions


def run_rho_k_sensitivity(exp_cfg: ExperimentConfig, 
                          rho_values: tuple = (0.0, 0.3, 0.5, 0.7, 1.0),
                          k_values: tuple = (0, 1, 2)) -> Dict:
    """Run sensitivity analysis over rho and K"""
    print("\n" + "="*80)
    print(f"SENSITIVITY ANALYSIS: rho={rho_values}, K={k_values}")
    print("="*80)
    
    # Build once so CSV mode derives dimensions before sensitivity loop
    cfg = build_config(exp_cfg)
    base_data = build_data(cfg)
    
    results = run_sensitivity_analysis(
        base_config=cfg,
        base_data=base_data,
        rho_list=rho_values,
        K_list=k_values,
        seed=42,
        show_plots=True,
        save_plots=cfg.save_plots,
        output_dir=cfg.plot_output_dir,
        n_jobs=exp_cfg.N_JOBS if exp_cfg.PARALLEL else 1,
    )
    return results


def run_multiple_solvers(exp_cfg: ExperimentConfig, 
                        solver_types: List[str]) -> Dict:
    """Run multiple solvers and compare results"""
    print("\n" + "="*80)
    print(f"MULTI-SOLVER BENCHMARK: {solver_types}")
    print("="*80)
    
    config = build_config(exp_cfg)
    data_ref = build_data(config)

    if exp_cfg.PARALLEL and len(solver_types) > 1:
        results = run_solver_types_and_collect(
            base_config=config,
            base_data=data_ref,
            solver_types=solver_types,
            seed=42,
            n_jobs=exp_cfg.N_JOBS,
        )
    else:
        solvers: List[BaseSolver] = []
        for solver_type in solver_types:
            cfg = copy.deepcopy(config)
            cfg.override(solver_type=solver_type)
            solvers.append(SolverFactory.create_solver(cfg, copy.deepcopy(data_ref)))
        results = run_solvers_and_collect(solvers)

    plot_comparison(results, save_plots=config.save_plots, output_dir=config.plot_output_dir)
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# QUICK START GUIDE FOR CUSTOM DATASETS:
# 
# Your CSV file must have columns: timestamp, server, dataset
# 
# 1. Built-in datasets (Netflix/Spotify):
#    exp = ExperimentConfig()
#    exp.DATASET_SOURCE = "netflix"  # or "spotify"
#    exp.CUSTOM_DATASET_PATH = "netflix_distilled.csv"  # or path to your file
#
# 2. Custom CSV files:
#    exp = ExperimentConfig()
#    exp.DATASET_SOURCE = "custom"
#    exp.CUSTOM_DATASET_PATH = "200netflix.csv"  # or any CSV with required columns
#
# 3. Synthetic (random) data:
#    exp = ExperimentConfig()
#    exp.DATASET_SOURCE = "synthetic"  # generates random data
#

if __name__ == "__main__":
    # ========== EXAMPLE 1: Single Solver Run ==========
    print("\n" + "#"*80)
    print("# EXAMPLE 1: Single Solver (MILP, Robust, CCG)")
    print("#"*80)
    
    exp1 = ExperimentConfig()
    exp1.T = 5
    exp1.NUM_DATASETS = 3
    exp1.NUM_SERVERS = 5
    exp1.K_FAILURES = 1
    exp1.RHO = 0.3
    exp1.SOLVER_TYPE = "milp" 
    exp1.USE_ROBUST = False
    exp1.USE_CCG = False
    exp1.VERBOSE = False
    exp1.PLOT_PER_SLOT = False
    exp1.PLOT_SUMMARY = True
    exp1.SAVE_OUTPUTS = True
    exp1.PARALLEL = True
    exp1.OUTPUT_DIR = f"results/run_single_solver/single_{exp1.SOLVER_TYPE}"
    exp1.DATASET_SOURCE = "csv"
    exp1.CSV_FILE = "netflix_distilled.csv"  
    
    solution1 = run_single_solver(exp1)
    
    # # ========== EXAMPLE 2: K-Failure Sweep ==========
    # print("\n" + "#"*80)
    # print("# EXAMPLE 2: K-Failure Sweep")
    # print("#"*80)
    
    # exp2 = ExperimentConfig()
    # exp2.T = 5
    # exp2.NUM_DATASETS = 3
    # exp2.NUM_SERVERS = 5
    # exp2.RHO = 0.5
    # exp2.SOLVER_TYPE = "milp"
    # exp2.VERBOSE = False
    
    # k_sweep = run_k_failure_sweep(exp2, k_values=[0, 1, 2])
    # print(f"✓ Completed K-sweep with {len(k_sweep)} configurations")
    
    # # ========== EXAMPLE 3: Solver Pair Comparison ==========
    # print("\n" + "#"*80)
    # print("# EXAMPLE 3: Solver Pair Comparison (MILP vs Greedy)")
    # print("#"*80)
    
    # exp3a = ExperimentConfig()
    # exp3a.T = 5
    # exp3a.SOLVER_TYPE = "milp"
    # exp3a.VERBOSE = False
    
    # exp3b = ExperimentConfig()
    # exp3b.T = 5
    # exp3b.SOLVER_TYPE = "custom"
    # exp3b.VERBOSE = False
    
    # comparison = run_solver_pair_comparison(exp3a, exp3b)
    # print("\nComparison Summary:")
    # for key, val in comparison.items():
    #     print(f"  {key}: {val}")
    
    # # ========== EXAMPLE 4: Multi-Solver Benchmark ==========
    # print("\n" + "#"*80)
    # print("# EXAMPLE 4: Multi-Solver Benchmark")
    # print("#"*80)
    
    # exp4 = ExperimentConfig()
    # exp4.T = 5
    # exp4.NUM_DATASETS = 3
    # exp4.NUM_SERVERS = 5
    # exp4.VERBOSE = False
    
    # multi_results = run_multiple_solvers(exp4, solver_types=["milp", "custom"])
    # print(f"✓ Benchmarked {len(multi_results)} solvers")
    
    # # ========== EXAMPLE 5: Custom Dataset (Netflix) ==========
    # print("\n" + "#"*80)
    # print("# EXAMPLE 5: Using Custom Dataset (Netflix CSV)")
    # print("#"*80)
    
    # # First validate the CSV format
    # if validate_csv_format("netflix_distilled.csv"):
    #     exp5 = ExperimentConfig()
    #     exp5.T = 10
    #     exp5.NUM_DATASETS = 3
    #     exp5.NUM_SERVERS = 5
    #     exp5.DATASET_SOURCE = "netflix"
    #     exp5.CUSTOM_DATASET_PATH = "netflix_distilled.csv"
    #     exp5.SOLVER_TYPE = "milp"
    #     exp5.VERBOSE = False
        
    #     solution5 = run_single_solver(exp5)
    #     print("✓ Netflix dataset loaded and solved successfully")
    
    # # ========== EXAMPLE 6: Custom Dataset Path (Any CSV) ==========
    # print("\n" + "#"*80)
    # print("# EXAMPLE 6: Using Custom CSV Format")
    # print("#"*80)
    
    # custom_csv = "200netflix.csv"
    # if validate_csv_format(custom_csv):
    #     exp6 = ExperimentConfig()
    #     exp6.T = 10
    #     exp6.NUM_DATASETS = 3
    #     exp6.NUM_SERVERS = 5
    #     exp6.DATASET_SOURCE = "custom"
    #     exp6.CUSTOM_DATASET_PATH = custom_csv
    #     exp6.SOLVER_TYPE = "custom"
    #     exp6.VERBOSE = False
        
    #     solution6 = run_single_solver(exp6)
    #     print(f"✓ Custom dataset {custom_csv} loaded and solved successfully")
    
    # # ========== EXAMPLE 7: ObjectiveCalculator Demo ==========
    # print("\n" + "#"*80)
    # print("# EXAMPLE 7: Centralized Objective Calculator")
    # print("#"*80)
    # print("This shows how to use the unified ObjectiveCalculator instead of")
    # print("having each solver compute objectives independently.\n")
    
    # exp7 = ExperimentConfig()
    # exp7.T = 3
    # exp7.NUM_DATASETS = 2
    # exp7.NUM_SERVERS = 4
    # exp7.K_FAILURES = 1
    # exp7.RHO = 0.5
    
    # demo_breakdown = demonstrate_objective_calculator(exp7)
    # print(f"✓ ObjectiveCalculator demo completed\n")
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
