from solver_factory import SolverFactory
from config import uEDDEConfig
from data_generator import DataGenerator
import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Dict, Any, List
from solver_base import BaseSolver, GlobalSolution, SlotSolution

def run_solvers_and_collect(
    solvers: List["BaseSolver"],   # instances (not classes)
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for solver in solvers:
        name = solver.__class__.__name__
        sol: "GlobalSolution" = solver.solve_all()

        # Aggregate totals & per-slot
        per_slot_obj, per_slot_R, per_slot_B, per_slot_Op = [], [], [], []
        per_slot_adds, per_slot_removes = [], []

        for slot in sol.slot_solutions:
            slot: "SlotSolution"
            per_slot_obj.append(float(slot.objective_value))
            per_slot_R.append(float(slot.R_nominal))
            per_slot_B.append(float(slot.B_nominal))
            per_slot_Op.append(float(slot.Op_cost))
            per_slot_adds.append(sum(slot.adds.values()) if slot.adds else 0)
            per_slot_removes.append(sum(slot.removes.values()) if slot.removes else 0)

        total_R = float(np.sum(per_slot_R))
        total_B = float(np.sum(per_slot_B))
        total_Op = float(np.sum(per_slot_Op))
        total_adds = int(np.sum(per_slot_adds))
        total_removes = int(np.sum(per_slot_removes))

        results[name] = {
            "solution": sol,
            "total": {
                "objective": float(sol.total_objective),
                "time": float(sol.total_solve_time),
                "R_total": total_R,
                "B_total": total_B,
                "Op_total": total_Op,
                "adds_total": total_adds,
                "removes_total": total_removes,
            },
            "per_slot": {
                "objective": per_slot_obj,
                "R": per_slot_R,
                "B": per_slot_B,
                "Op": per_slot_Op,
                "adds": per_slot_adds,
                "removes": per_slot_removes,
            }
        }

    return results


def _freeze_data_snapshot(data: DataGenerator) -> Dict[str, Any]:
    return {
        "initial_state": dict(data.initial_state),
        "counts": dict(data.counts),
        "weights_nominal": dict(data.weights_nominal),
        "weights_error": dict(data.weights_error),
        "active_datasets": dict(data.active_datasets),
        "attachment_points": {t: list(aps) for t, aps in data.attachment_points.items()},
    }


def _build_data_from_snapshot(cfg: uEDDEConfig, frozen: Dict[str, Any], seed: int) -> DataGenerator:
    data = DataGenerator(cfg, seed=seed)
    data.initial_state = dict(frozen["initial_state"])
    data.counts = dict(frozen["counts"])
    data.weights_nominal = dict(frozen["weights_nominal"])
    data.weights_error = dict(frozen["weights_error"])
    data.active_datasets = dict(frozen["active_datasets"])
    data.attachment_points = {t: list(v) for t, v in frozen["attachment_points"].items()}
    return data


def _collect_solution_stats(sol: GlobalSolution) -> Dict[str, Any]:
    per_slot_obj, per_slot_R, per_slot_B, per_slot_Op = [], [], [], []
    per_slot_adds, per_slot_removes = [], []
    for slot in sol.slot_solutions:
        per_slot_obj.append(float(slot.objective_value))
        per_slot_R.append(float(slot.R_nominal))
        per_slot_B.append(float(slot.B_nominal))
        per_slot_Op.append(float(slot.Op_cost))
        per_slot_adds.append(sum(slot.adds.values()) if slot.adds else 0)
        per_slot_removes.append(sum(slot.removes.values()) if slot.removes else 0)

    return {
        "total": {
            "objective": float(sol.total_objective),
            "time": float(sol.total_solve_time),
            "R_total": float(np.sum(per_slot_R)),
            "B_total": float(np.sum(per_slot_B)),
            "Op_total": float(np.sum(per_slot_Op)),
            "adds_total": int(np.sum(per_slot_adds)),
            "removes_total": int(np.sum(per_slot_removes)),
        },
        "per_slot": {
            "objective": per_slot_obj,
            "R": per_slot_R,
            "B": per_slot_B,
            "Op": per_slot_Op,
            "adds": per_slot_adds,
            "removes": per_slot_removes,
        },
    }


def _solver_type_worker(base_config: uEDDEConfig, frozen: Dict[str, Any], solver_type: str, seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.override(solver_type=solver_type, plot_per_slot=False, verbose=False)
    data = _build_data_from_snapshot(cfg, frozen, seed)
    solver = SolverFactory.create_solver(cfg, data)
    sol = solver.solve_all()
    name = solver.__class__.__name__
    payload = _collect_solution_stats(sol)
    payload["solution"] = sol
    return {"name": name, "payload": payload}


def run_solver_types_and_collect(
    base_config: uEDDEConfig,
    base_data: DataGenerator,
    solver_types: List[str],
    seed: int = 42,
    n_jobs: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Parallel-friendly benchmark entrypoint using solver type names."""
    if not solver_types:
        return {}

    frozen = _freeze_data_snapshot(base_data)
    if n_jobs == 1 or len(solver_types) == 1:
        results = {}
        for solver_type in solver_types:
            out = _solver_type_worker(base_config, frozen, solver_type, seed)
            results[out["name"]] = out["payload"]
        return results

    max_workers = n_jobs if n_jobs and n_jobs > 0 else min(len(solver_types), os.cpu_count() or 1)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_solver_type_worker, base_config, frozen, solver_type, seed): solver_type
            for solver_type in solver_types
        }
        for fut in as_completed(futures):
            out = fut.result()
            results[out["name"]] = out["payload"]
    return results


def _k_worker(base_config: uEDDEConfig, frozen: Dict[str, Any], solver_type: str, k_value: int, seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.override(solver_type=solver_type, K_failures=int(k_value), plot_per_slot=False, verbose=False)
    data = _build_data_from_snapshot(cfg, frozen, seed)
    solver = SolverFactory.create_solver(cfg, data)
    sol = solver.solve_all()
    return {"K": int(k_value), "solution": sol}


def run_k_sweep_parallel(
    base_config: uEDDEConfig,
    base_data: DataGenerator,
    k_values: List[int],
    solver_type: str,
    seed: int = 42,
    n_jobs: int = 0,
) -> Dict[int, GlobalSolution]:
    """Run K-sweep in parallel for a fixed solver type over frozen data."""
    if not k_values:
        return {}

    frozen = _freeze_data_snapshot(base_data)
    if n_jobs == 1 or len(k_values) == 1:
        out = {}
        for k_value in k_values:
            row = _k_worker(base_config, frozen, solver_type, int(k_value), seed)
            out[row["K"]] = row["solution"]
        return out

    max_workers = n_jobs if n_jobs and n_jobs > 0 else min(len(k_values), os.cpu_count() or 1)
    out = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_k_worker, base_config, frozen, solver_type, int(k_value), seed): int(k_value)
            for k_value in k_values
        }
        for fut in as_completed(futures):
            row = fut.result()
            out[row["K"]] = row["solution"]
    return out


def plot_comparison(results: Dict[str, Dict[str, Any]], save_plots: bool = False, output_dir: str = "results"):
    """
    Make a compact comparison:
      1) Grouped bars for TOTAL objective, R_total, B_total, Op_total.
      2) Lines (multi-series) for per-slot objective across solvers.
      3) Lines (multi-series) for per-slot adds/removes across solvers.
    """
    solver_names = list(results.keys())
    n = len(solver_names)

    out_dir = Path(output_dir)
    if save_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    def emit(fig, suffix: str):
        if save_plots:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"{ts}_{suffix}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {path}")
        else:
            plt.show()

    # ========== 1) Grouped bar chart for totals ==========
    totals_obj = [results[nm]["total"]["objective"] for nm in solver_names]
    totals_R   = [results[nm]["total"]["R_total"] for nm in solver_names]
    totals_B   = [results[nm]["total"]["B_total"] for nm in solver_names]
    totals_Op  = [results[nm]["total"]["Op_total"] for nm in solver_names]

    x = np.arange(n)
    width = 0.2

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - 1.5*width, totals_obj, width, label="Objective")
    ax1.bar(x - 0.5*width, totals_R,   width, label="R_total")
    ax1.bar(x + 0.5*width, totals_B,   width, label="B_total")
    ax1.bar(x + 1.5*width, totals_Op,  width, label="Op_total")

    ax1.set_xticks(x)
    ax1.set_xticklabels(solver_names, rotation=15)
    ax1.set_ylabel("Total value")
    ax1.set_title("Totals: Objective vs R/B/Op")
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # ========== 2) Per-slot objective line plot ==========
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    max_T = 0
    for nm in solver_names:
        per_slot_obj = results[nm]["per_slot"]["objective"]
        max_T = max(max_T, len(per_slot_obj))
        ax2.plot(range(1, len(per_slot_obj)+1), per_slot_obj, marker='o', label=nm)

    ax2.set_xlabel("Time slot t")
    ax2.set_ylabel("Objective (per-slot)")
    ax2.set_title("Per-slot Objective Comparison")
    ax2.set_xticks(range(1, max_T+1))
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ========== 3) Per-slot adds/removes line plots ==========
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    max_T = 0
    for nm in solver_names:
        adds = results[nm]["per_slot"]["adds"]
        rems = results[nm]["per_slot"]["removes"]
        max_T = max(max_T, len(adds))
        ax3a.plot(range(1, len(adds)+1), adds, marker='s', label=nm)
        ax3b.plot(range(1, len(rems)+1), rems, marker='^', label=nm)

    ax3a.set_title("Per-slot Adds")
    ax3a.set_xlabel("Time slot t")
    ax3a.set_ylabel("#Adds")
    ax3a.set_xticks(range(1, max_T+1))
    ax3a.grid(True, alpha=0.3)
    ax3a.legend()

    ax3b.set_title("Per-slot Removes")
    ax3b.set_xlabel("Time slot t")
    ax3b.set_ylabel("#Removes")
    ax3b.set_xticks(range(1, max_T+1))
    ax3b.grid(True, alpha=0.3)
    ax3b.legend()

    plt.tight_layout()
    emit(fig1, "benchmark_totals")
    emit(fig2, "benchmark_per_slot_objective")
    emit(fig3, "benchmark_adds_removes")


# ============================
# Example usage (adjust this)
# ============================
if __name__ == "__main__":
    NUM_DATASETS = 3
    NUM_SERVERS = 5
    NUM_TIME_SLOTS = 5

    server_capacities = {j: 100000 for j in range(NUM_SERVERS)}
    hop_budgets = {i: 1 for i in range(NUM_DATASETS)}
    dataset_sizes = {i: 1 for i in range(NUM_DATASETS)}

    base = uEDDEConfig()
    base.override(
        T=NUM_TIME_SLOTS,
        num_datasets=NUM_DATASETS,
        num_servers=NUM_SERVERS,
        rho=0.3,
        K_failures=0,
        Gamma_budget=1.5,
        lambda_add=1.0,
        solver_time_limit=10_000,   
        plot_per_slot=False,
        server_capacities=server_capacities,
        hop_budgets=hop_budgets,
        dataset_sizes=dataset_sizes,
        dataset_source="synthetic",
    )

    # Generate data ONCE for fair comparison
    data = DataGenerator(base, seed=42)
    data.generate_all(activity=1)

    # Create two solver-specific configs that only differ by algorithm settings
    cfg_milp = copy.deepcopy(base)
    cfg_milp.override(use_robust=True, use_ccg=True, solver_type="milp")

    cfg_greedy = copy.deepcopy(base)
    cfg_greedy.override(use_robust=False, use_ccg=False, solver_type="greedy")

    # Reuse the SAME generated data for both solvers (deep-copied for safety)
    data_milp = copy.deepcopy(data)
    data_greedy = copy.deepcopy(data)

    milp_solver = SolverFactory.create_solver(cfg_milp, data_milp)
    greedy_solver = SolverFactory.create_solver(cfg_greedy, data_greedy)

    solvers_to_run: List["BaseSolver"] = [
        milp_solver,
        greedy_solver,
    ]

    # Safety check
    if not solvers_to_run:
        print("Please populate `solvers_to_run` with your solver instances.")
    else:
        results = run_solvers_and_collect(solvers_to_run)
        plot_comparison(results)
