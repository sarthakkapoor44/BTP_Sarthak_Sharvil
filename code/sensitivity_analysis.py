import copy
from typing import Iterable
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd

from config import uEDDEConfig
from data_generator import DataGenerator
from solver_factory import SolverFactory


def _run_single_sensitivity_point(
    base_config: uEDDEConfig,
    frozen: dict,
    seed: int,
    rho: float,
    k_value: int,
) -> dict:
    """Worker: solve one (rho, K) point and return aggregate metrics."""
    cfg = copy.deepcopy(base_config)
    cfg.override(
        rho=float(rho),
        K_failures=int(k_value),
        plot_per_slot=False,
        verbose=False,
    )
    cfg.hop_distances = dict(frozen["hop_distances"])

    data = DataGenerator(cfg, seed=seed)
    data.initial_state = dict(frozen["initial_state"])
    data.counts = dict(frozen["counts"])
    data.weights_nominal = dict(frozen["weights_nominal"])
    data.weights_error = dict(frozen["weights_error"])
    data.active_datasets = dict(frozen["active_datasets"])
    data.attachment_points = {t: list(v) for t, v in frozen["attachment_points"].items()}

    solver = SolverFactory.create_solver(cfg, data)
    sol = solver.solve_all()

    avg_R_nom = np.mean([s.R_nominal for s in sol.slot_solutions]) if sol.slot_solutions else 0.0
    avg_B_nom = np.mean([s.B_nominal for s in sol.slot_solutions]) if sol.slot_solutions else 0.0
    tot_cost = float(sum(s.Op_cost for s in sol.slot_solutions))
    feas_slots = sum(1 for s in sol.slot_solutions if s.status in ("Optimal", "Feasible"))

    return {
        "rho": float(rho),
        "K": int(k_value),
        "Gamma": cfg.Gamma_budget,
        "T": cfg.T,
        "total_obj": float(sol.total_objective),
        "avg_R_nom": float(avg_R_nom),
        "avg_B_nom": float(avg_B_nom),
        "total_cost": float(tot_cost),
        "solve_time": float(sol.total_solve_time),
        "feasible_slots": int(feas_slots),
    }

def run_sensitivity_analysis(
    base_config: uEDDEConfig,
    base_data: DataGenerator,
    rho_list: Iterable[float] = (0.0, 0.3, 0.5, 0.7, 1.0),
    K_list: Iterable[int] = (0, 1, 2),
    seed: int = 42,
    show_plots: bool = True,
    save_plots: bool = False,
    output_dir: str = "results",
    n_jobs: int = 0,
) -> pd.DataFrame:
    """Run joint sensitivity analysis on rho and K using fixed, prebuilt data."""
    results = []

    num_servers = base_config.num_servers

    # Freeze everything so every (rho, K) run uses identical data
    frozen = {
        "initial_state": dict(base_data.initial_state),
        "counts": dict(base_data.counts),
        "weights_nominal": dict(base_data.weights_nominal),
        "weights_error": dict(base_data.weights_error),
        "active_datasets": dict(base_data.active_datasets),
        "attachment_points": {t: list(aps) for t, aps in base_data.attachment_points.items()},
        "hop_distances": dict(base_config.hop_distances),
    }

    verbose = bool(getattr(base_config, "verbose", True))
    if verbose:
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS: Joint grid over (rho, K)")
        print("="*80)

    jobs = []
    for K in K_list:
        if K > num_servers:
            if verbose:
                print(f"Skipping K={K} (exceeds num_servers={num_servers}).")
            continue
        for rho in rho_list:
            jobs.append((float(rho), int(K)))

    if not jobs:
        return pd.DataFrame(columns=["rho", "K", "Gamma", "T", "total_obj", "avg_R_nom", "avg_B_nom", "total_cost", "solve_time", "feasible_slots"])

    if n_jobs == 1 or len(jobs) == 1:
        for rho, k_value in jobs:
            if verbose:
                print(f"\n--- Testing (rho={rho}, K={k_value}) ---")
            results.append(_run_single_sensitivity_point(base_config, frozen, seed, rho, k_value))
    else:
        max_workers = n_jobs if n_jobs and n_jobs > 0 else min(len(jobs), os.cpu_count() or 1)
        if verbose:
            print(f"Running {len(jobs)} sensitivity jobs in parallel with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_sensitivity_point, base_config, frozen, seed, rho, k_value): (rho, k_value)
                for rho, k_value in jobs
            }
            for fut in as_completed(futures):
                rho, k_value = futures[fut]
                if verbose:
                    print(f"Completed (rho={rho}, K={k_value})")
                results.append(fut.result())

    # ---------------------------------
    # Results table
    # ---------------------------------
    df = pd.DataFrame(results).sort_values(["K", "rho"]).reset_index(drop=True)
    if verbose:
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS RESULTS (grid over (rho, K))")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

    if save_plots:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"{ts}_sensitivity_grid.csv"
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"Saved sensitivity results: {csv_path}")

    # ---------------------------------
    # Line plots: K on x; one line per rho
    # ---------------------------------
    if show_plots and len(df):
        import matplotlib.pyplot as plt

        metrics = [
            ("total_obj",  "Total Objective"),
            ("avg_B_nom",  "Avg Benefit (nominal)"),
            ("total_cost", "Migration Cost"),
            ("solve_time", "Solve Time (s)")
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        K_vals   = sorted(df["K"].unique())
        rho_vals = sorted(df["rho"].unique())

        for ax, (metric, ylabel) in zip(axes, metrics):
            for rho in rho_vals:
                sub = df[df["rho"] == rho]
                # Ensure values align to all K on x-axis (NaN if missing)
                y = [sub.loc[sub["K"] == K, metric].values[0] if (sub["K"] == K).any() else np.nan
                     for K in K_vals]
                ax.plot(K_vals, y, marker="o", label=f"ρ={rho:g}")
            ax.set_title(f"{ylabel} vs K", fontweight="bold")
            ax.set_xlabel("K failures", fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_xticks(K_vals)
            ax.grid(True, alpha=0.3)
            ax.legend(title="rho", fontsize=9, ncol=2)

        plt.tight_layout()
        if save_plots:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = out_dir / f"{ts}_sensitivity_plots.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if verbose:
                print(f"Saved plot: {fig_path}")
        else:
            plt.show()

    return df
