import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from solver_base import BaseSolver, GlobalSolution
from config import uEDDEConfig
from data_generator import DataGenerator

# -----------------------------------------------------------------------------
# Utility: running time average
# -----------------------------------------------------------------------------
def running_average(values: List[float]) -> List[float]:
    avg = []
    s = 0.0
    for i, v in enumerate(values, start=1):
        s += v
        avg.append(s / i)
    return avg


# -----------------------------------------------------------------------------
# Run a solver and extract metrics
# -----------------------------------------------------------------------------
def run_solver(solver: BaseSolver) -> Dict:
    """
    Runs a solver using its solve_all() method and extracts
    comparable metrics.
    """
    wall_start = time.time()
    global_sol: GlobalSolution = solver.solve_all()
    wall_end = time.time()

    slot_solutions = global_sol.slot_solutions
    T = len(slot_solutions)

    per_slot_obj = [s.objective_value for s in slot_solutions]
    time_avg_obj = running_average(per_slot_obj)

    adds = sum(len(s.adds) for s in slot_solutions)
    removes = sum(len(s.removes) for s in slot_solutions)

    return {
        "name": solver.__class__.__name__,
        "slot_solutions": slot_solutions,
        "per_slot_objective": per_slot_obj,
        "time_avg_objective": time_avg_obj,
        "total_objective": global_sol.total_objective,
        "avg_objective": float(np.mean(per_slot_obj)) if T > 0 else 0.0,
        "total_runtime": wall_end - wall_start,
        "avg_runtime_per_slot": global_sol.total_solve_time / T if T > 0 else 0.0,
        "adds": adds,
        "removes": removes,
    }


# -----------------------------------------------------------------------------
# Constraint violation checks
# -----------------------------------------------------------------------------
def count_coverage_violations(slot_solutions, config, data) -> int:
    """
    Counts uncovered (i,p,t) pairs.
    """
    violations = 0
    for sol in slot_solutions:
        t = sol.time_slot
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
                    violations += 1
    return violations


def count_capacity_violations(slot_solutions, config) -> int:
    """
    Counts server capacity violations.
    """
    violations = 0
    for sol in slot_solutions:
        for j in range(config.num_servers):
            used = sum(
                config.dataset_sizes[i]
                for i in range(config.num_datasets)
                if sol.states.get((i, j), 0) == 1
            )
            if used > config.server_capacities[j]:
                violations += 1
    return violations


# -----------------------------------------------------------------------------
# Main comparison function
# -----------------------------------------------------------------------------
def compare_two_solvers(
    solver_A: BaseSolver,
    solver_B: BaseSolver,
    config: uEDDEConfig,
    data: DataGenerator,
    plot: bool = True
) -> Dict:
    """
    Fair online comparison of two solvers.
    """

    print("\n" + "=" * 80)
    print(f"COMPARING SOLVERS")
    print(f"{solver_A.__class__.__name__}  vs  {solver_B.__class__.__name__}")
    print("=" * 80)

    res_A = run_solver(solver_A)
    res_B = run_solver(solver_B)

    cov_A = count_coverage_violations(res_A["slot_solutions"], config, data)
    cov_B = count_coverage_violations(res_B["slot_solutions"], config, data)

    cap_A = count_capacity_violations(res_A["slot_solutions"], config)
    cap_B = count_capacity_violations(res_B["slot_solutions"], config)

    summary = {
        "Solver A": res_A["name"],
        "Solver B": res_B["name"],

        "Total Objective A": res_A["total_objective"],
        "Total Objective B": res_B["total_objective"],

        "Avg Objective A": res_A["avg_objective"],
        "Avg Objective B": res_B["avg_objective"],

        "Avg Runtime/Slot A": res_A["avg_runtime_per_slot"],
        "Avg Runtime/Slot B": res_B["avg_runtime_per_slot"],

        "Total Runtime A": res_A["total_runtime"],
        "Total Runtime B": res_B["total_runtime"],

        "Adds A": res_A["adds"],
        "Adds B": res_B["adds"],

        "Removes A": res_A["removes"],
        "Removes B": res_B["removes"],

        "Coverage Violations A": cov_A,
        "Coverage Violations B": cov_B,

        "Capacity Violations A": cap_A,
        "Capacity Violations B": cap_B,
    }

    # Print summary
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Optional plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(res_A["time_avg_objective"], label=res_A["name"])
        plt.plot(res_B["time_avg_objective"], label=res_B["name"])
        plt.xlabel("Time Slot")
        plt.ylabel("Time-Average Objective")
        plt.title("Time-Average Objective Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return summary
