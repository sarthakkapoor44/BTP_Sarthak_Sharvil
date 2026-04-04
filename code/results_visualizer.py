from solver_factory import GlobalSolution
from data_generator import DataGenerator
from config import uEDDEConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

class ResultsVisualizer:
    """
    Comprehensive visualization for uEDDE results
    Supports per-slot details and aggregate analysis
    """
    
    def __init__(self, solution: GlobalSolution, data_gen: DataGenerator):
        self.solution = solution
        self.data = data_gen
        self.config = solution.config
        self.save_plots = bool(getattr(self.config, "save_plots", False))
        self.output_dir = Path(getattr(self.config, "plot_output_dir", "results"))
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _emit_figure(self, fig, name: str):
        if self.save_plots:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self.output_dir / f"{ts}_{name}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if self.config.verbose:
                print(f"Saved plot: {out}")
        else:
            plt.show()
    
    def plot_all(self, k_sweep_results: Optional[List[Tuple[int, float]]] = None):
        """Generate all plots"""
        if self.config.verbose:
            print("\n" + "="*80)
            print("GENERATING VISUALIZATIONS")
            print("="*80)
        
        self.plot_objective_over_time()
        self.plot_objective_vs_k_failures(k_sweep_results)
        self.plot_objective_vs_requests()
        self.plot_placement_evolution()
        self.plot_objective_components()
        self.plot_migration_costs()
        self.plot_network_topology()
        
        if self.config.use_robust:
            self.plot_robust_comparison()
        
        if self.config.verbose:
            print("✓ All visualizations complete")
    
    def plot_objective_over_time(self):
        """Plot objective value over time slots"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = [sol.time_slot for sol in self.solution.slot_solutions]
        objectives = [sol.objective_value for sol in self.solution.slot_solutions]
        
        ax.plot(times, objectives, marker='o', linewidth=2, markersize=10, 
               label='Per-Slot Objective', color='#2E86AB')
        ax.axhline(y=np.mean(objectives), color='red', linestyle='--', 
                  label=f'Mean={np.mean(objectives):.3f}', alpha=0.7)
        
        ax.set_xlabel('Time Slot', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
        ax.set_title('Objective Value Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._emit_figure(fig, "objective_over_time")
    
    def plot_objective_vs_k_failures(self, k_sweep_results: Optional[List[Tuple[int, float]]] = None):
        """Plot objective sensitivity to K_failures"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        if k_sweep_results:
            k_sweep_sorted = sorted(k_sweep_results, key=lambda kv: kv[0])
            k_values = [kv[0] for kv in k_sweep_sorted]
            objectives = [kv[1] for kv in k_sweep_sorted]
            
            ax.plot(k_values, objectives, marker='o', linewidth=2, markersize=10,
                    label='Total Objective', color='#2E86AB')
            
            for k_val, obj in zip(k_values, objectives):
                ax.annotate(f'K={k_val}', (k_val, obj), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=9)
            
            ax.set_title('Objective vs. Failure Budget K (sweep results)',
                         fontsize=14, fontweight='bold')
        else:
            if self.config.verbose:
                print("  Note: Provide k_sweep_results to visualize actual K sweep. Using current run only.")
            k_current = self.config.K_failures
            obj_current = self.solution.total_objective
            ax.scatter([k_current], [obj_current], s=200, color='red', 
                       label=f'Current K={k_current}', zorder=5)
            
            k_range = range(0, min(5, self.config.num_servers))
            expected_decay = [obj_current * (1 - 0.15 * k) for k in k_range]
            ax.plot(k_range, expected_decay, '--', alpha=0.5, color='gray', 
                    label='Expected Trend (illustrative)')
            ax.set_title('Objective vs. Failure Budget K', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('K (Failure Budget)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Objective', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._emit_figure(fig, "objective_vs_k_failures")
    
    def plot_objective_vs_requests(self):
        """Plot objective vs total requests per slot"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        requests_per_slot = []
        objectives = []
        
        for sol in self.solution.slot_solutions:
            t = sol.time_slot
            total_requests = sum(self.data.counts.get((i,p,t), 0) 
                                for i in range(self.config.num_datasets) 
                                for p in self.data.attachment_points.get(t, []))
            requests_per_slot.append(total_requests)
            objectives.append(sol.objective_value)
        
        # Scatter plot
        ax.scatter(requests_per_slot, objectives, s=150, alpha=0.7, color='#A23B72')
        
        # Add trend line
        if len(requests_per_slot) > 1:
            z = np.polyfit(requests_per_slot, objectives, 1)
            p = np.poly1d(z)
            ax.plot(sorted(requests_per_slot), p(sorted(requests_per_slot)), 
                   "--", alpha=0.5, color='black', label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        
        # Annotate points
        for i, (req, obj) in enumerate(zip(requests_per_slot, objectives)):
            ax.annotate(f't={i+1}', (req, obj), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        
        ax.set_xlabel('Total Requests', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
        ax.set_title('Objective vs. Total Requests per Slot', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._emit_figure(fig, "objective_vs_requests")
    
    def plot_placement_evolution(self):
        """Plot placement evolution (heatmap over time)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Total replicas per dataset
        replicas_per_dataset = np.zeros((self.config.num_datasets, self.config.T))
        for sol in self.solution.slot_solutions:
            t = sol.time_slot - 1
            for i in range(self.config.num_datasets):
                count = sum(1 for (ii,j), val in sol.states.items() if ii == i and val > 0)
                replicas_per_dataset[i, t] = count
        
        im1 = axes[0].imshow(replicas_per_dataset, cmap='YlOrRd', aspect='auto')
        axes[0].set_xlabel('Time Slot', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Dataset', fontsize=12, fontweight='bold')
        axes[0].set_title('Number of Replicas per Dataset Over Time', fontsize=13, fontweight='bold')
        axes[0].set_xticks(range(self.config.T))
        axes[0].set_xticklabels(range(1, self.config.T+1))
        axes[0].set_yticks(range(self.config.num_datasets))
        plt.colorbar(im1, ax=axes[0], label='Replica Count')
        
        # Adds and removes per slot
        adds_per_slot = [len(sol.adds) for sol in self.solution.slot_solutions]
        removes_per_slot = [len(sol.removes) for sol in self.solution.slot_solutions]
        
        x = np.arange(1, self.config.T + 1)
        width = 0.35
        axes[1].bar(x - width/2, adds_per_slot, width, label='Adds', color='#06A77D', alpha=0.8)
        axes[1].bar(x + width/2, removes_per_slot, width, label='Removes', color='#D72638', alpha=0.8)
        axes[1].set_xlabel('Time Slot', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1].set_title('Migration Operations per Time Slot', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._emit_figure(fig, "placement_evolution")
    
    def plot_objective_components(self):
        """Plot objective components breakdown"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        times = [sol.time_slot for sol in self.solution.slot_solutions]
        R_nom = [sol.R_nominal for sol in self.solution.slot_solutions]
        B_nom = [sol.B_nominal for sol in self.solution.slot_solutions]
        Op = [-sol.Op_cost for sol in self.solution.slot_solutions]  # Negative for cost
        
        width = 0.25
        x = np.array(times)
        
        ax.bar(x - width, R_nom, width, label='R_nominal (Dedup)', color='#4ECDC4', alpha=0.9)
        ax.bar(x, B_nom, width, label='B_nominal (Benefit)', color='#FF6B6B', alpha=0.9)
        ax.bar(x + width, Op, width, label='Op_cost (negative)', color='#95E1D3', alpha=0.9)
        
        if self.config.use_robust:
            R_wc = [sol.R_wc for sol in self.solution.slot_solutions]
            B_wc = [sol.B_wc for sol in self.solution.slot_solutions]
            ax.plot(x, R_wc, 'o--', label='R_wc', color='#1A535C', markersize=6, linewidth=2)
            ax.plot(x, B_wc, 's--', label='B_wc', color='#F25C54', markersize=6, linewidth=2)
        
        ax.set_xlabel('Time Slot', fontsize=12, fontweight='bold')
        ax.set_ylabel('Component Value', fontsize=12, fontweight='bold')
        ax.set_title('Objective Components Breakdown', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, ncol=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._emit_figure(fig, "objective_components")
    
    def plot_migration_costs(self):
        """Plot migration costs over time"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = [sol.time_slot for sol in self.solution.slot_solutions]
        costs = [sol.Op_cost for sol in self.solution.slot_solutions]
        cumulative = np.cumsum(costs)
        
        ax.bar(times, costs, color='#F18F01', alpha=0.7, label='Per-Slot Cost')
        ax.plot(times, cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative Cost')
        
        ax.set_xlabel('Time Slot', fontsize=12, fontweight='bold')
        ax.set_ylabel('Migration Cost', fontsize=12, fontweight='bold')
        ax.set_title('Migration Costs Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._emit_figure(fig, "migration_costs")
    
    def plot_robust_comparison(self):
        """Compare nominal vs worst-case performance"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        times = [sol.time_slot for sol in self.solution.slot_solutions]
        
        # Dedup ratio
        R_nom = [sol.R_nominal for sol in self.solution.slot_solutions]
        R_wc = [sol.R_wc for sol in self.solution.slot_solutions]
        
        axes[0].plot(times, R_nom, 'o-', label='R_nominal', linewidth=2, markersize=8, color='#06A77D')
        axes[0].plot(times, R_wc, 's--', label='R_wc', linewidth=2, markersize=8, color='#D72638')
        axes[0].fill_between(times, R_nom, R_wc, alpha=0.2, color='gray')
        axes[0].set_ylabel('Dedup Ratio', fontsize=12, fontweight='bold')
        axes[0].set_title('Nominal vs Worst-Case Dedup Ratio', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Benefit
        B_nom = [sol.B_nominal for sol in self.solution.slot_solutions]
        B_wc = [sol.B_wc for sol in self.solution.slot_solutions]
        
        axes[1].plot(times, B_nom, 'o-', label='B_nominal', linewidth=2, markersize=8, color='#2E86AB')
        axes[1].plot(times, B_wc, 's--', label='B_wc', linewidth=2, markersize=8, color='#A23B72')
        axes[1].fill_between(times, B_nom, B_wc, alpha=0.2, color='gray')
        axes[1].set_xlabel('Time Slot', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Benefit', fontsize=12, fontweight='bold')
        axes[1].set_title('Nominal vs Worst-Case Benefit', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._emit_figure(fig, "robust_comparison")
    
    def plot_network_topology(self):
        """Plot and save network topology visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Build graph from adjacency list
        G = nx.Graph()
        adjacency = self.config.adjacency
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        # Use spring layout for nice positioning
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=800, alpha=0.9, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, ax=ax)
        
        # Compute network statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        diameter = nx.diameter(G) if nx.is_connected(G) else float('inf')
        
        # Add statistics text
        info_text = f"Nodes: {num_nodes} | Edges: {num_edges} | Avg Degree: {avg_degree:.2f} | Diameter: {diameter}"
        fig.text(0.5, 0.02, info_text, ha='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=11)
        
        ax.set_title('Network Topology', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        self._emit_figure(fig, "network_topology")
    
    def save_network_topology_info(self):
        """Save adjacency list and hop distances to file"""
        if not self.save_plots:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        topology_path = self.output_dir / f"{ts}_network_topology.txt"
        
        with topology_path.open("w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("NETWORK TOPOLOGY INFORMATION\n")
            f.write("="*80 + "\n\n")
            
            # Adjacency list
            f.write("ADJACENCY LIST:\n")
            f.write("-"*80 + "\n")
            for node, neighbors in sorted(self.config.adjacency.items()):
                f.write(f"Server {node}: {neighbors}\n")
            
            # Network statistics
            f.write("\n" + "="*80 + "\n")
            f.write("NETWORK STATISTICS:\n")
            f.write("-"*80 + "\n")
            
            G = nx.Graph()
            for node, neighbors in self.config.adjacency.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
            diameter = nx.diameter(G) if nx.is_connected(G) else float('inf')
            
            f.write(f"Number of servers (nodes): {num_nodes}\n")
            f.write(f"Number of links (edges): {num_edges}\n")
            f.write(f"Average degree: {avg_degree:.2f}\n")
            f.write(f"Network diameter: {diameter}\n")
            
            # Hop distances matrix
            f.write("\n" + "="*80 + "\n")
            f.write("HOP DISTANCES (source → target):\n")
            f.write("-"*80 + "\n")
            f.write("From each server to each server:\n\n")
            
            for src in range(self.config.num_servers):
                dists = []
                for dst in range(self.config.num_servers):
                    dist = self.config.hop_distances.get((src, dst), float('inf'))
                    if dist == float('inf'):
                        dists.append(f"{dst}:∞")
                    else:
                        dists.append(f"{dst}:{int(dist)}")
                f.write(f"Server {src}: {', '.join(dists)}\n")
            
            # Hop budgets for datasets
            f.write("\n" + "="*80 + "\n")
            f.write("HOP BUDGETS (per dataset):\n")
            f.write("-"*80 + "\n")
            for dataset_id, hop_budget in sorted(self.config.hop_budgets.items()):
                f.write(f"Dataset {dataset_id}: {hop_budget} hop(s)\n")
            
            f.write("\n" + "="*80 + "\n")
        
        if self.config.verbose:
            print(f"Saved network topology info: {topology_path}")

    def save_coverage_violations_report(self):
        """Save a slot-by-slot coverage violation report to file."""
        if not self.save_plots:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_csv_path = self.output_dir / f"{ts}_coverage_violations.csv"
        violations_txt_path = self.output_dir / f"{ts}_coverage_violations.txt"

        violation_rows = []
        slot_summary = []

        for sol in self.solution.slot_solutions:
            t = sol.time_slot
            state = sol.states
            slot_violations = 0
            slot_demand_pairs = 0

            for i in range(self.config.num_datasets):
                if self.data.active_datasets.get((i, t), 0) != 1:
                    continue

                H_i = self.config.hop_budgets[i]
                for p in self.data.attachment_points.get(t, []):
                    request_count = self.data.counts.get((i, p, t), 0)
                    if request_count <= 0:
                        continue

                    slot_demand_pairs += 1
                    min_hop = float("inf")
                    for j in range(self.config.num_servers):
                        if state.get((i, j), 0) != 1:
                            continue
                        hop = self.config.hop_distances.get((j, p), float("inf"))
                        if hop < min_hop:
                            min_hop = hop

                    if min_hop > H_i:
                        slot_violations += 1
                        violation_rows.append(
                            {
                                "slot": t,
                                "dataset": i,
                                "attachment_point": p,
                                "requests": request_count,
                                "hop_budget": H_i,
                                "min_hop": "inf" if min_hop == float("inf") else int(min_hop),
                                "reason": "uncovered_within_hop_budget",
                            }
                        )

            slot_summary.append(
                {
                    "slot": t,
                    "demand_pairs": slot_demand_pairs,
                    "violations": slot_violations,
                    "violation_ratio": (slot_violations / slot_demand_pairs) if slot_demand_pairs else 0.0,
                }
            )

        violations_df = pd.DataFrame(
            violation_rows,
            columns=["slot", "dataset", "attachment_point", "requests", "hop_budget", "min_hop", "reason"],
        )
        violations_df.to_csv(violations_csv_path, index=False)

        summary_df = pd.DataFrame(slot_summary)
        total_violations = len(violation_rows)
        total_demand_pairs = int(summary_df["demand_pairs"].sum()) if not summary_df.empty else 0
        overall_ratio = (total_violations / total_demand_pairs) if total_demand_pairs else 0.0

        with violations_txt_path.open("w", encoding="utf-8") as f:
            f.write("COVERAGE VIOLATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total demand pairs checked: {total_demand_pairs}\n")
            f.write(f"Total violations: {total_violations}\n")
            f.write(f"Overall violation ratio: {overall_ratio:.6f}\n\n")
            f.write("Per-slot summary:\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n")
            if total_violations > 0:
                f.write("\nFirst violations:\n")
                f.write(violations_df.head(50).to_string(index=False))
                f.write("\n")

        if self.config.verbose:
            print(f"Saved coverage violations: {violations_csv_path}")
            print(f"Saved coverage violation summary: {violations_txt_path}")
    
    def print_statistics_table(self, save_to_file: bool = False):
        """Print summary statistics table"""
        data = []
        for sol in self.solution.slot_solutions:
            data.append({
                'Slot': sol.time_slot,
                'Objective': f"{sol.objective_value:.4f}",
                'R_nom': f"{sol.R_nominal:.4f}",
                'B_nom': f"{sol.B_nominal:.4f}",
                'R_wc': f"{sol.R_wc:.4f}" if self.config.use_robust else "N/A",
                'B_wc': f"{sol.B_wc:.4f}" if self.config.use_robust else "N/A",
                'Op': f"{sol.Op_cost:.4f}",
                'Adds': len(sol.adds),
                'Removes': len(sol.removes),
                'Time(s)': f"{sol.solve_time:.2f}"
            })
        
        df = pd.DataFrame(data)
        if self.config.verbose:
            print("\n" + "="*80)
            print("SUMMARY STATISTICS TABLE")
            print("="*80)
            print(df.to_string(index=False))
            print("="*80)
            print(f"TOTAL OBJECTIVE: {self.solution.total_objective:.4f}")
            print(f"TOTAL TIME: {self.solution.total_solve_time:.2f}s")
            print("="*80)

        if save_to_file or self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"{ts}_summary_statistics.csv"
            # txt_path = self.output_dir / f"{ts}_summary_statistics.txt"
            actions_csv_path = self.output_dir / f"{ts}_actions.csv"
            placement_csv_path = self.output_dir / f"{ts}_placement_timeline.csv"
            # actions_txt_path = self.output_dir / f"{ts}_actions.txt"
            df.to_csv(csv_path, index=False)
            # with txt_path.open("w", encoding="utf-8") as f:
            #     f.write(df.to_string(index=False))
            #     f.write("\n")
            #     f.write(f"TOTAL OBJECTIVE: {self.solution.total_objective:.4f}\n")
            #     f.write(f"TOTAL TIME: {self.solution.total_solve_time:.2f}s\n")

            action_rows = []
            for sol in self.solution.slot_solutions:
                for (dataset_id, server_id), val in sorted(sol.adds.items()):
                    if val > 0:
                        action_rows.append({
                            "Slot": sol.time_slot,
                            "Action": "ADD",
                            "Dataset": dataset_id,
                            "Server": server_id,
                        })
                for (dataset_id, server_id), val in sorted(sol.removes.items()):
                    if val > 0:
                        action_rows.append({
                            "Slot": sol.time_slot,
                            "Action": "REMOVE",
                            "Dataset": dataset_id,
                            "Server": server_id,
                        })

            actions_df = pd.DataFrame(action_rows, columns=["Slot", "Action", "Dataset", "Server"])
            actions_df.to_csv(actions_csv_path, index=False)

            # Placement timeline: one initial row, then one row per slot after model decision.
            placement_rows = [{
                "time_step": 0,
                "slot": 0,
                "phase": "initial_before_any_decision",
                "actions": "-",
                "placement": self._format_placement(self.data.initial_state),
            }]

            for sol in self.solution.slot_solutions:
                action_texts = []
                for (dataset_id, server_id), val in sorted(sol.adds.items()):
                    if val > 0:
                        action_texts.append(f"ADD:{dataset_id},{server_id}")
                for (dataset_id, server_id), val in sorted(sol.removes.items()):
                    if val > 0:
                        action_texts.append(f"REMOVE:{dataset_id},{server_id}")

                placement_rows.append({
                    "time_step": int(sol.time_slot),
                    "slot": int(sol.time_slot),
                    "phase": "after_slot_decision",
                    "actions": ";".join(action_texts) if action_texts else "-",
                    "placement": self._format_placement(sol.states),
                })

            placement_df = pd.DataFrame(
                placement_rows,
                columns=["time_step", "slot", "phase", "actions", "placement"],
            )
            placement_df.to_csv(placement_csv_path, index=False)

            # with actions_txt_path.open("w", encoding="utf-8") as f:
            #     if action_rows:
            #         f.write("REPLICA ACTIONS (add/remove by slot, dataset, server)\n")
            #         f.write(actions_df.to_string(index=False))
            #         f.write("\n")
            #         f.write(f"TOTAL_ACTIONS: {len(action_rows)}\n")
            #         f.write(f"TOTAL_ADDS: {sum(1 for r in action_rows if r['Action'] == 'ADD')}\n")
            #         f.write(f"TOTAL_REMOVES: {sum(1 for r in action_rows if r['Action'] == 'REMOVE')}\n")
            #     else:
            #         f.write("No add/remove actions were performed in this run.\n")

            if self.config.verbose:
                print(f"Saved stats: {csv_path}")
                # print(f"Saved stats: {txt_path}")
                print(f"Saved actions: {actions_csv_path}")
                print(f"Saved placement timeline: {placement_csv_path}")
                # print(f"Saved actions: {actions_txt_path}")
            
            # Save network topology info
            self.save_network_topology_info()
            self.save_coverage_violations_report()

    def _format_placement(self, state):
        rows = []
        for j in range(self.config.num_servers):
            datasets = [str(i) for i in range(self.config.num_datasets) if state.get((i, j), 0) == 1]
            datasets_text = ",".join(datasets) if datasets else "-"
            rows.append(f"server {j}: {datasets_text}")
        return " | ".join(rows)

