from solver_factory import GlobalSolution
from data_generator import DataGenerator
from config import uEDDEConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
                # print(f"Saved actions: {actions_txt_path}")

