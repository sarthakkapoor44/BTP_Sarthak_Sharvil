#!/usr/bin/env python3
"""Visualize RL actions against demand, hop limits, and availability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import ExperimentConfig, build_config, build_data


def _latest_actions_csv(search_root: Path) -> Optional[Path]:
    candidates = list(search_root.rglob("*_actions.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_workspace_config(dataset_path: Path):
    exp = ExperimentConfig()
    exp.DATASET_SOURCE = "csv"
    exp.CSV_FILE = str(dataset_path)
    exp.VERBOSE = False
    exp.PLOT_PER_SLOT = False
    exp.PLOT_SUMMARY = False
    exp.SAVE_OUTPUTS = False
    config = build_config(exp)
    data = build_data(config)
    return config, data


def _coverage_matrix(config, data, state: Dict[Tuple[int, int], int], t: int, P_t: List[int]):
    matrix = {}
    for i in range(config.num_datasets):
        H_i = config.hop_budgets[i]
        for p in P_t:
            req = data.counts.get((i, p, t), 0)
            if req <= 0:
                matrix[(i, p)] = {
                    "requests": 0,
                    "covered": False,
                    "best_hop": None,
                }
                continue

            best_hop = None
            for j in range(config.num_servers):
                if state.get((i, j), 0) != 1:
                    continue
                hop = config.hop_distances.get((j, p), float("inf"))
                if hop <= H_i and (best_hop is None or hop < best_hop):
                    best_hop = hop

            matrix[(i, p)] = {
                "requests": req,
                "covered": best_hop is not None,
                "best_hop": best_hop,
            }
    return matrix


def _coverage_stats(matrix):
    demand_pairs = [v for v in matrix.values() if v["requests"] > 0]
    total_pairs = len(demand_pairs)
    covered_pairs = sum(1 for v in demand_pairs if v["covered"])
    total_requests = sum(v["requests"] for v in demand_pairs)
    covered_requests = sum(v["requests"] for v in demand_pairs if v["covered"])
    return {
        "demand_pairs": total_pairs,
        "covered_pairs": covered_pairs,
        "uncovered_pairs": total_pairs - covered_pairs,
        "total_requests": total_requests,
        "covered_requests": covered_requests,
        "coverage_ratio_pairs": covered_pairs / total_pairs if total_pairs else 0.0,
        "coverage_ratio_requests": covered_requests / total_requests if total_requests else 0.0,
    }


def _action_details(config, data, t: int, before: Dict[Tuple[int, int], int], after: Dict[Tuple[int, int], int], actions: pd.DataFrame):
    P_t = data.attachment_points.get(t, [])
    before_cov = _coverage_matrix(config, data, before, t, P_t)
    after_cov = _coverage_matrix(config, data, after, t, P_t)
    before_stats = _coverage_stats(before_cov)
    after_stats = _coverage_stats(after_cov)

    rows = []
    for _, row in actions.iterrows():
        action_type = str(row["Action"]).upper()
        i = int(row["Dataset"])
        j = int(row["Server"])
        active = data.active_datasets.get((i, t), 0) == 1
        size_i = config.dataset_sizes[i]
        load_before = sum(config.dataset_sizes[k] * before.get((k, j), 0) for k in range(config.num_datasets))
        load_after = sum(config.dataset_sizes[k] * after.get((k, j), 0) for k in range(config.num_datasets))
        cap_ok_after = load_after <= config.server_capacities[j] + 1e-9
        served_aps = []
        served_requests = 0
        for p in P_t:
            req = data.counts.get((i, p, t), 0)
            if req <= 0:
                continue
            if config.hop_distances.get((j, p), float("inf")) <= config.hop_budgets[i]:
                served_aps.append(p)
                served_requests += req

        if action_type == "ADD":
            duplicate = before.get((i, j), 0) == 1
            reason_parts = []
            reason_parts.append("active" if active else "inactive")
            reason_parts.append("new" if not duplicate else "duplicate")
            reason_parts.append("cap_ok" if cap_ok_after else "cap_violation")
            reason_parts.append("serves_demand" if served_requests > 0 else "no_demand_in_hop_range")
            feasible = active and (not duplicate) and cap_ok_after and served_requests > 0
            status = "OK" if feasible else "INVALID"
        else:
            replica_exists = before.get((i, j), 0) == 1
            temp = dict(before)
            temp[(i, j)] = 0
            removed_cov = _coverage_matrix(config, data, temp, t, P_t)
            removed_stats = _coverage_stats(removed_cov)
            preserves_coverage = removed_stats["uncovered_pairs"] == 0
            reason_parts = []
            reason_parts.append("exists" if replica_exists else "missing")
            reason_parts.append("coverage_ok" if preserves_coverage else "coverage_break")
            status = "OK" if replica_exists and preserves_coverage else "INVALID"

        rows.append({
            "Action": action_type,
            "Dataset": i,
            "Server": j,
            "Status": status,
            "Reason": ", ".join(reason_parts),
            "Served_APs": ", ".join(map(str, served_aps)) if served_aps else "-",
            "Served_Requests": served_requests,
            "Load_Before": load_before,
            "Load_After": load_after,
            "Cap_OK_After": "Y" if cap_ok_after else "N",
        })

    return before_cov, after_cov, before_stats, after_stats, pd.DataFrame(rows)


def _render_slot_dashboard(config, data, t: int, before_state: Dict[Tuple[int, int], int], after_state: Dict[Tuple[int, int], int], actions: pd.DataFrame, output_dir: Path):
    P_t = data.attachment_points.get(t, [])
    if not P_t:
        return None

    before_cov, after_cov, before_stats, after_stats, action_rows = _action_details(
        config, data, t, before_state, after_state, actions
    )

    datasets = list(range(config.num_datasets))
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[4.2, 3.0], height_ratios=[3.2, 1.8], wspace=0.25, hspace=0.35)

    def _heatmap(ax, coverage_map, title):
        mat = np.zeros((config.num_datasets, len(P_t)))
        for r, i in enumerate(datasets):
            for c, p in enumerate(P_t):
                mat[r, c] = data.counts.get((i, p, t), 0)

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Attachment point / server")
        ax.set_ylabel("Dataset")
        ax.set_xticks(range(len(P_t)))
        ax.set_xticklabels(P_t)
        ax.set_yticks(range(config.num_datasets))
        ax.set_yticklabels(datasets)

        for r, i in enumerate(datasets):
            for c, p in enumerate(P_t):
                req = data.counts.get((i, p, t), 0)
                if req <= 0:
                    continue
                covered = coverage_map[(i, p)]["covered"]
                label = f"{req}\n{'Y' if covered else 'N'}"
                color = "black" if req < mat.max() * 0.6 else "white"
                ax.text(c, r, label, ha="center", va="center", fontsize=8, color=color)
                edge = "#1b9e77" if covered else "#d95f02"
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=edge, linewidth=2)
                ax.add_patch(rect)

        plt.colorbar(im, ax=ax, label="Request count")

    ax_before = fig.add_subplot(grid[0, 0])
    _heatmap(ax_before, before_cov, f"Slot {t}: demand before actions")

    ax_after = fig.add_subplot(grid[0, 1])
    _heatmap(ax_after, after_cov, f"Slot {t}: demand after actions")

    ax_table = fig.add_subplot(grid[1, 0])
    ax_table.axis("off")
    if len(action_rows) == 0:
        ax_table.text(0.0, 0.8, "No actions in this slot", fontsize=12, fontweight="bold")
    else:
        table_df = action_rows[["Action", "Dataset", "Server", "Status", "Reason", "Served_APs"]].copy()
        table_df = table_df.head(8)
        tbl = ax_table.table(
            cellText=table_df.values,
            colLabels=table_df.columns.tolist(),
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.4)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                status = table_df.iloc[r - 1]["Status"]
                cell.set_facecolor("#e8f5e9" if status == "OK" else "#ffebee")
    ax_table.set_title("Actions and constraint check", fontsize=12, fontweight="bold")

    ax_summary = fig.add_subplot(grid[1, 1])
    ax_summary.axis("off")
    summary_text = (
        f"Total requests: {before_stats['total_requests']}\n"
        f"Demand pairs: {before_stats['demand_pairs']}\n"
        f"Covered pairs before: {before_stats['covered_pairs']}\n"
        f"Covered pairs after: {after_stats['covered_pairs']}\n"
        f"Uncovered pairs after: {after_stats['uncovered_pairs']}\n"
        f"Coverage ratio after: {after_stats['coverage_ratio_pairs']:.3f}\n"
        f"Actions: {len(action_rows)}\n"
        f"Adds: {(action_rows['Action'] == 'ADD').sum() if len(action_rows) else 0}\n"
        f"Removes: {(action_rows['Action'] == 'REMOVE').sum() if len(action_rows) else 0}"
    )
    ax_summary.text(0.02, 0.95, summary_text, va="top", fontsize=11, family="monospace")

    out_path = output_dir / f"slot_{t:04d}_action_dashboard.png"
    fig.suptitle(f"Action trace dashboard for slot {t}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "slot": t,
        "requests": before_stats["total_requests"],
        "covered_pairs_before": before_stats["covered_pairs"],
        "covered_pairs_after": after_stats["covered_pairs"],
        "uncovered_pairs_after": after_stats["uncovered_pairs"],
        "coverage_ratio_after": after_stats["coverage_ratio_pairs"],
        "actions": len(action_rows),
        "adds": int((action_rows["Action"] == "ADD").sum()) if len(action_rows) else 0,
        "removes": int((action_rows["Action"] == "REMOVE").sum()) if len(action_rows) else 0,
        "dashboard_path": str(out_path),
        "action_rows": action_rows,
    }


def _render_overview(summary_df: pd.DataFrame, output_dir: Path):
    if summary_df.empty:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)

    axes[0].plot(summary_df["slot"], summary_df["coverage_ratio_after"], color="#1b9e77", linewidth=2)
    axes[0].set_ylabel("Coverage ratio")
    axes[0].set_title("Coverage ratio after actions over time")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(summary_df["slot"] - 0.15, summary_df["adds"], width=0.3, label="Adds", color="#2ecc71")
    axes[1].bar(summary_df["slot"] + 0.15, summary_df["removes"], width=0.3, label="Removes", color="#e74c3c")
    axes[1].set_ylabel("Action count")
    axes[1].set_title("Actions taken per slot")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].plot(summary_df["slot"], summary_df["uncovered_pairs_after"], color="#d95f02", linewidth=2, label="Uncovered demand pairs")
    axes[2].plot(summary_df["slot"], summary_df["covered_pairs_after"], color="#1f78b4", linewidth=2, label="Covered demand pairs")
    axes[2].set_xlabel("Time slot")
    axes[2].set_ylabel("Demand pairs")
    axes[2].set_title("Coverage status over time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    out_path = output_dir / "action_trace_overview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _compact_state_string(state: Dict[Tuple[int, int], int], num_datasets: int, num_servers: int) -> str:
    """Serialize placement as a compact dataset->server list."""
    rows = []
    for i in range(num_datasets):
        servers = [str(j) for j in range(num_servers) if state.get((i, j), 0) == 1]
        rows.append(f"{i}:{'|'.join(servers) if servers else '-'}")
    return ";".join(rows)


def _demand_string(data, t: int, num_datasets: int) -> str:
    """Serialize all demands for a slot as dataset@server=count entries."""
    entries = []
    for i in range(num_datasets):
        for p in data.attachment_points.get(t, []):
            req = int(data.counts.get((i, p, t), 0))
            if req > 0:
                entries.append(f"{i}@{p}={req}")
    return ";".join(entries) if entries else "-"


def export_trace_csv(dataset_path: Path, actions_path: Path, output_csv: Path):
    """Create a long-form CSV with one row per time slot."""
    config, data = _load_workspace_config(dataset_path)
    actions_df = pd.read_csv(actions_path)

    state = dict(data.initial_state)
    rows = []

    for t in range(1, config.T + 1):
        slot_actions = actions_df[actions_df["Slot"] == t].copy()
        before_state = dict(state)
        before_cov = _coverage_matrix(config, data, before_state, t, data.attachment_points.get(t, []))
        before_stats = _coverage_stats(before_cov)

        action_texts = []
        action_statuses = []
        for _, row in slot_actions.iterrows():
            action = str(row["Action"]).upper()
            i = int(row["Dataset"])
            j = int(row["Server"])
            prev = state.get((i, j), 0)
            if action == "ADD":
                state[(i, j)] = 1
            elif action == "REMOVE":
                state[(i, j)] = 0
            action_texts.append(f"{action}:{i},{j}")
            action_statuses.append("1" if prev != state.get((i, j), 0) else "0")

        after_state = dict(state)
        after_cov = _coverage_matrix(config, data, after_state, t, data.attachment_points.get(t, []))
        after_stats = _coverage_stats(after_cov)

        rows.append({
            "slot": t,
            "timestamp": t - 1,
            "action_count": int(len(slot_actions)),
            "actions": ";".join(action_texts) if action_texts else "-",
            "actions_applied": ";".join(action_statuses) if action_statuses else "-",
            "placement_before": _compact_state_string(before_state, config.num_datasets, config.num_servers),
            "placement_after": _compact_state_string(after_state, config.num_datasets, config.num_servers),
            "demand": _demand_string(data, t - 1, config.num_datasets),
            "attachment_points": ",".join(map(str, data.attachment_points.get(t, []))) if data.attachment_points.get(t, []) else "-",
            "total_requests": before_stats["total_requests"],
            "demand_pairs": before_stats["demand_pairs"],
            "covered_pairs_before": before_stats["covered_pairs"],
            "covered_pairs_after": after_stats["covered_pairs"],
            "uncovered_pairs_after": after_stats["uncovered_pairs"],
            "coverage_ratio_after": after_stats["coverage_ratio_pairs"],
        })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def reconstruct_and_visualize(dataset_path: Path, actions_path: Path, output_dir: Path, max_detail_slots: int = 12):
    config, data = _load_workspace_config(dataset_path)
    actions_df = pd.read_csv(actions_path)
    if actions_df.empty:
        raise ValueError(f"No actions found in {actions_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    state = dict(data.initial_state)
    summary_rows = []

    action_slots = []
    for t in range(1, config.T + 1):
        slot_actions = actions_df[actions_df["Slot"] == t]
        before_state = dict(state)

        for _, row in slot_actions.iterrows():
            action = str(row["Action"]).upper()
            i = int(row["Dataset"])
            j = int(row["Server"])
            if action == "ADD":
                state[(i, j)] = 1
            elif action == "REMOVE":
                state[(i, j)] = 0

        after_state = dict(state)
        slot_summary = _render_slot_dashboard(config, data, int(t), before_state, after_state, slot_actions, output_dir)
        if slot_summary is not None:
            summary_rows.append({k: v for k, v in slot_summary.items() if k != "action_rows"})
            if len(slot_actions) > 0:
                action_slots.append(int(t))

    summary_df = pd.DataFrame(summary_rows).sort_values("slot") if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_csv = output_dir / "action_trace_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        overview_path = _render_overview(summary_df, output_dir)
    else:
        summary_csv = None
        overview_path = None

    if len(action_slots) > max_detail_slots:
        action_slots = action_slots[:max_detail_slots]

    return {
        "summary_csv": summary_csv,
        "overview_path": overview_path,
        "action_slots": action_slots,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize RL action traces against demand and constraints.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to netflix_distilled.csv or similar CSV.")
    parser.add_argument("--actions", type=Path, default=None, help="Path to *_actions.csv generated by ResultsVisualizer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/action_trace_visuals"), help="Directory for generated figures.")
    parser.add_argument("--trace-csv", type=Path, default=None, help="Optional output CSV path for the slot-level trace.")
    parser.add_argument("--max-detail-slots", type=int, default=12, help="Maximum detail dashboards to generate.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset or (repo_root / "netflix_distilled.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    actions_path = args.actions
    if actions_path is None:
        actions_path = _latest_actions_csv(repo_root / "results")
        if actions_path is None:
            raise FileNotFoundError("Could not locate an actions CSV. Pass --actions explicitly.")

    trace_csv = args.trace_csv or (args.output_dir / "action_trace.csv")
    trace_csv = export_trace_csv(dataset_path, actions_path, trace_csv)
    report = reconstruct_and_visualize(dataset_path, actions_path, args.output_dir, max_detail_slots=args.max_detail_slots)
    print(f"Saved trace CSV: {trace_csv}")
    print(f"Saved overview: {report['overview_path']}")
    print(f"Saved summary CSV: {report['summary_csv']}")
    print(f"Saved detailed dashboards for slots: {report['action_slots']}")


if __name__ == "__main__":
    main()