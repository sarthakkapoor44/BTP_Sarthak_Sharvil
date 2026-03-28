import copy
import pandas as pd
from solver_base import GlobalSolution
from config import uEDDEConfig

def _as_indexable_capacity(cap, J):
    """Allow server_capacities as list OR dict."""
    if isinstance(cap, dict):
        return {int(k): float(v) for k, v in cap.items()}
    return {j: float(cap[j]) for j in range(J)}

def extract_migration_df(
    global_sol: "GlobalSolution",
    cfg: "uEDDEConfig",
    name: str,
    op_cost_stored_as: str,   # "raw" for custom, "effective" for milp
) -> pd.DataFrame:
    """
    Build a migration-cost dataframe with BOTH:
      - op_cost_raw       : unscaled add-cost (comparable across solvers)
      - op_cost_effective : scaled cost used in objective (lambda_add * raw)

    IMPORTANT:
      - MILP typically stores Op_cost as *effective* (already includes lambda_add).
      - Custom typically stores Op_cost as *raw* and scales only in objective.
    """
    assert op_cost_stored_as in ("raw", "effective")

    rows = []
    lam = float(getattr(cfg, "lambda_add", 1.0))
    lam_safe = lam if abs(lam) > 1e-12 else 1.0  # avoid div-by-zero

    for slot in global_sol.slot_solutions:
        t = int(slot.time_slot)
        op_stored = float(slot.Op_cost)

        if op_cost_stored_as == "raw":
            op_raw = op_stored
            op_effective = op_raw * lam
        else:  # "effective"
            op_effective = op_stored
            op_raw = op_effective / lam_safe

        adds_cnt = int(sum(slot.adds.values())) if getattr(slot, "adds", None) else 0
        rems_cnt = int(sum(slot.removes.values())) if getattr(slot, "removes", None) else 0

        rows.append({
            "solver": name,
            "time_slot": t,
            "op_cost_stored": op_stored,
            "op_cost_stored_as": op_cost_stored_as,
            "op_cost_raw": op_raw,
            "op_cost_effective": op_effective,
            "adds": adds_cnt,
            "removes": rems_cnt,
        })

    df = pd.DataFrame(rows).sort_values("time_slot").reset_index(drop=True)
    df["cum_op_cost_raw"] = df["op_cost_raw"].cumsum()
    df["cum_op_cost_effective"] = df["op_cost_effective"].cumsum()
    return df

import matplotlib.pyplot as plt

def plot_migration_costs_separate(
    df_milp: pd.DataFrame,
    df_custom: pd.DataFrame,
    use_raw: bool = False,
    milp_color: str = "red",
    custom_color: str = "blue",
    milp_linestyle: str = "--",   # dashed
    custom_linestyle: str = "-",  # solid
    legend_fontsize: int = 12,
    label_fontsize: int = 10,
    border_lw: float = 1.2,
):
    y = "op_cost_raw" if use_raw else "op_cost_effective"
    ycum = "cum_op_cost_raw" if use_raw else "cum_op_cost_effective"
    suffix = "(raw)" if use_raw else "(effective)"

    def _add_border(ax):
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(border_lw)
            sp.set_color("black")

    # ---------- Plot 1: Per-slot ----------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    _add_border(ax1)

    ax1.plot(df_milp["time_slot"], df_milp[y],
             color=milp_color, linestyle=milp_linestyle, linewidth=2, marker="o",
             label=f"MILP {suffix}")
    ax1.plot(df_custom["time_slot"], df_custom[y],
             color=custom_color, linestyle=custom_linestyle, linewidth=2, marker="s",
             label=f"Custom {suffix}")

    ax1.set_title(f"Per-slot Migration Cost {suffix}", fontsize=label_fontsize + 2)
    ax1.set_xlabel("Time slot", fontsize=label_fontsize)
    ax1.set_ylabel("Migration cost", fontsize=label_fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=legend_fontsize)
    ax1.tick_params(axis="both", labelsize=label_fontsize)
    fig1.tight_layout()
    plt.show()

    # ---------- Plot 2: Cumulative ----------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    _add_border(ax2)

    ax2.plot(df_milp["time_slot"], df_milp[ycum],
             color=milp_color, linestyle=milp_linestyle, linewidth=2, marker="o",
             label=f"MILP cumulative {suffix}")
    ax2.plot(df_custom["time_slot"], df_custom[ycum],
             color=custom_color, linestyle=custom_linestyle, linewidth=2, marker="s",
             label=f"Custom cumulative {suffix}")

    ax2.set_title(f"Cumulative Migration Cost {suffix}", fontsize=label_fontsize + 2)
    ax2.set_xlabel("Time slot", fontsize=label_fontsize)
    ax2.set_ylabel("Cumulative cost", fontsize=label_fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=legend_fontsize)
    ax2.tick_params(axis="both", labelsize=label_fontsize)
    fig2.tight_layout()
    plt.show()


