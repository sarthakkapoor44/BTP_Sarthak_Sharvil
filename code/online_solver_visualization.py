import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from solver_base import BaseSolver, GlobalSolution

# -----------------------------------------------------------------------------
# Utility: running average
# -----------------------------------------------------------------------------
def running_average(values: List[float]) -> List[float]:
    avg = []
    s = 0.0
    for i, v in enumerate(values, start=1):
        s += v
        avg.append(s / i)
    return avg


# -----------------------------------------------------------------------------
# Run solver and collect metrics
# -----------------------------------------------------------------------------
def run_solver(solver: BaseSolver) -> Dict:
    """
    Runs a solver and extracts all metrics needed for comparison and plots.
    """
    wall_start = time.time()
    global_solution: GlobalSolution = solver.solve_all()
    wall_end = time.time()

    slots = global_solution.slot_solutions
    T = len(slots)

    per_slot_obj = [s.objective_value for s in slots]
    time_avg_obj = running_average(per_slot_obj)
    cumulative_obj = np.cumsum(per_slot_obj)

    return {
        "name": solver.__class__.__name__,
        "slot_solutions": slots,
        "per_slot_objective": per_slot_obj,
        "time_avg_objective": time_avg_obj,
        "cumulative_objective": cumulative_obj,
        "total_objective": global_solution.total_objective,
        "avg_objective": float(np.mean(per_slot_obj)) if T > 0 else 0.0,
        "total_runtime": wall_end - wall_start,
        "avg_runtime_per_slot": global_solution.total_solve_time / T if T > 0 else 0.0,
        "adds": sum(len(s.adds) for s in slots),
        "removes": sum(len(s.removes) for s in slots),
    }


# -----------------------------------------------------------------------------
# Constraint violation metrics
# -----------------------------------------------------------------------------
def coverage_violations_over_time(results, config, data):
    viol = []
    for sol in results["slot_solutions"]:
        t = sol.time_slot
        count = 0
        for i in range(config.num_datasets):
            if data.active_datasets.get((i, t), 0) != 1:
                continue
            H_i = config.hop_budgets[i]
            for p in data.attachment_points.get(t, []):
                if data.counts.get((i, p, t), 0) <= 0:
                    continue
                neigh = data.get_neighborhood(p, H_i)
                covered = any(sol.states.get((i, j), 0) == 1 for j in neigh)
                if not covered:
                    count += 1
        viol.append(count)
    return viol


# -----------------------------------------------------------------------------
# ========================= VISUALIZATIONS ====================================
# -----------------------------------------------------------------------------

def _emit_or_show(fig, save_plots: bool, output_dir: str, suffix: str):
    if save_plots:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{ts}_{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")
    else:
        plt.show()


def plot_per_slot_objective(results_dict, save_plots: bool = False, output_dir: str = "results"):
    plt.figure(figsize=(10, 4))
    for name, res in results_dict.items():
        plt.plot(res["per_slot_objective"], label=name, alpha=0.7)
    plt.xlabel("Time Slot")
    plt.ylabel("Per-Slot Objective")
    plt.title("Per-Slot Objective (Online Behavior)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _emit_or_show(plt.gcf(), save_plots, output_dir, "online_per_slot_objective")


def plot_time_average_objective(results_dict, save_plots: bool = False, output_dir: str = "results"):
    plt.figure(figsize=(10, 4))
    for name, res in results_dict.items():
        plt.plot(res["time_avg_objective"], label=name)
    plt.xlabel("Time Slot")
    plt.ylabel("Time-Average Objective")
    plt.title("Time-Average Objective (Main Performance Metric)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _emit_or_show(plt.gcf(), save_plots, output_dir, "online_time_average_objective")


def plot_cumulative_objective(results_dict, save_plots: bool = False, output_dir: str = "results"):
    plt.figure(figsize=(10, 4))
    for name, res in results_dict.items():
        plt.plot(res["cumulative_objective"], label=name)
    plt.xlabel("Time Slot")
    plt.ylabel("Cumulative Objective")
    plt.title("Cumulative Objective (Long-Term Gain)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _emit_or_show(plt.gcf(), save_plots, output_dir, "online_cumulative_objective")


def plot_migration_activity(results_dict, save_plots: bool = False, output_dir: str = "results"):
    plt.figure(figsize=(10, 4))
    for name, res in results_dict.items():
        adds = [len(s.adds) for s in res["slot_solutions"]]
        removes = [len(s.removes) for s in res["slot_solutions"]]
        plt.plot(adds, linestyle="--", label=f"{name} adds")
        plt.plot(removes, linestyle=":", label=f"{name} removes")
    plt.xlabel("Time Slot")
    plt.ylabel("# Actions")
    plt.title("Migration Activity (Explains Objective Dips)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _emit_or_show(plt.gcf(), save_plots, output_dir, "online_migration_activity")


def plot_coverage_violations(results_dict, config, data, save_plots: bool = False, output_dir: str = "results"):
    plt.figure(figsize=(10, 4))
    for name, res in results_dict.items():
        viol = coverage_violations_over_time(res, config, data)
        plt.plot(viol, label=name)
    plt.xlabel("Time Slot")
    plt.ylabel("# Coverage Violations")
    plt.title("Coverage Violations Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _emit_or_show(plt.gcf(), save_plots, output_dir, "online_coverage_violations")


# -----------------------------------------------------------------------------
# ========================= MASTER DRIVER ======================================
# -----------------------------------------------------------------------------
def visualize_solvers(solvers: Dict[str, BaseSolver], config, data):
    """
    Runs solvers, collects metrics, and generates all standard plots.
    """
    save_plots = bool(getattr(config, "save_plots", False))
    output_dir = str(getattr(config, "plot_output_dir", "results"))

    results = {}
    for name, solver in solvers.items():
        print(f"\nRunning {name}...")
        results[name] = run_solver(solver)

    plot_per_slot_objective(results, save_plots=save_plots, output_dir=output_dir)
    plot_time_average_objective(results, save_plots=save_plots, output_dir=output_dir)
    plot_cumulative_objective(results, save_plots=save_plots, output_dir=output_dir)
    plot_migration_activity(results, save_plots=save_plots, output_dir=output_dir)
    plot_coverage_violations(results, config, data, save_plots=save_plots, output_dir=output_dir)

    print("\nSUMMARY:")
    rows = []
    for name, res in results.items():
        line = (
            f"{name:12s} | "
            f"AvgObj={res['avg_objective']:.4f} | "
            f"Adds={res['adds']:4d} | "
            f"Removes={res['removes']:4d} | "
            f"AvgTime/slot={res['avg_runtime_per_slot']:.4f}s"
        )
        print(line)
        rows.append({
            "solver": name,
            "avg_objective": res["avg_objective"],
            "adds": res["adds"],
            "removes": res["removes"],
            "avg_runtime_per_slot": res["avg_runtime_per_slot"],
        })

    if save_plots:
        import pandas as pd
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"{ts}_online_visualization_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

    return results
