from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from graphGen import generate_random_connected_adjacency,generate_random_connected_adjacency,visualize_adjacency


@dataclass
class uEDDEConfig:
    """
    Complete configuration for Online uEDDE with Two-Stage Robust Optimization
    All parameters can be manually overridden
    """
    
    # ========== TIME HORIZON ==========
    T: int = 5  # Number of time slots
    
    # ========== DATASETS ==========
    num_datasets: int = 3
    dataset_sizes: Dict[int, float] = field(default_factory=lambda: {})  # s_i
    hop_budgets: Dict[int, int] = field(default_factory=lambda: {})  # H_i (dataset-specific)
    
    # ========== SERVERS ==========
    num_servers: int = 5
    server_capacities: Dict[int, float] = field(default_factory=lambda: {})  # C_j
    server_bandwidth: Dict[int, float] = field(default_factory=lambda: {})  # BW_j (optional ingress)
    
    # ========== NETWORK TOPOLOGY ==========
    adjacency: Dict[int, List[int]] = field(default_factory=lambda: {})  # Adjacency list
    hop_distances: Dict[Tuple[int, int], int] = field(default_factory=lambda: {})  # Precomputed h_{j,p}
    
    # ========== ROBUSTNESS PARAMETERS ==========
    use_robust: bool = True  # Enable two-stage robust optimization
    use_repair_in_robust: bool = True  # Apply K+1 repair in robust mode; if False, only evaluate worst-case on solver output
    K_failures: int = 1  # Maximum failures per slot (K_t)
    Gamma_budget: float = 2.0  # Bertsimas-Sim uncertainty budget
    use_k_survivable_coverage: bool = False  # Use m_{ip,t} = min(K_t+1, |N_p(H_i)|) in nominal coverage
    
    # ========== OPTIMIZATION WEIGHTS ==========
    rho: float = 0.5  # Trade-off: 0=nominal only, 1=worst-case only
    lambda_add: float = 1.0  # Weight for add (migration) cost
    eta_stability: float = 0.0  # Optional stability penalty for state flips
    
    # ========== OPTIONAL CONSTRAINTS ==========
    use_ingress_constraint: bool = False  # Enforce bandwidth limits
    use_xor_constraint: bool = True  # Prevent simultaneous add/remove
    enforce_recourse_from_prev: bool = False  # q_{ij,t} <= A_{ij,t-1} (paper-faithful)
    
    # ========== SOLVER SELECTION ==========
    solver_type: str = "milp"  # Options: "milp", "greedy_adr", "custom"
    solver_time_limit: int = 300  # Time limit in seconds for MILP
    solver_gap: float = 0.01  # MIP gap tolerance
    
    # ========== CCG (Column-and-Constraint Generation) ==========
    use_ccg: bool = True  # Use iterative CCG for exact solution
    ccg_max_iterations: int = 20  # Max CCG iterations per time slot
    ccg_tolerance: float = 1e-4  # Convergence tolerance
    
    # ========== GREEDY HEURISTIC PARAMETERS ==========
    greedy_enable_pruning: bool = True  # Reverse-prune redundant replicas
    greedy_scoring_method: str = "robust"  # Options: "nominal", "robust", "hybrid"
    greedy_bandit_enabled: bool = True  # Enable online bandit refinement on top of greedy
    greedy_bandit_iters: int = 20  # Local-search steps per time slot
    greedy_bandit_alpha: float = 0.65  # Blend: learned reward vs one-step objective delta
    greedy_bandit_ucb_c: float = 0.35  # UCB exploration bonus scale
    greedy_bandit_temperature: float = 0.02  # Simulated-annealing acceptance for small downhill moves

    # ========== KMEANS DATASET SOLVER PARAMETERS ==========
    kmeans_k_search: str = "binary"  # Options: "binary", "linear"

    # ========== Q-LEARNING PARAMETERS ==========
    q_pretrain_episodes: int = 0  # Offline warm-start episodes before online solve
    q_pretrain_max_steps: int = 0  # 0 means solver-derived default
    q_pretrain_epsilon: float = 0.35  # Exploration used only during pretraining
    q_epsilon_decay: float = 0.995  # Per-slot epsilon decay after decisions
    q_epsilon_min: float = 0.01  # Lower bound for epsilon decay
    q_learn_feasibility: bool = True  # Learn a soft feasibility mask from experience
    q_feasibility_threshold: float = 0.55  # Skip candidates predicted below this probability
    q_feasibility_lr: float = 0.05  # Online learning rate for feasibility predictor
    q_feasibility_min_trials: int = 5  # Require some history before masking aggressively

    # ========== BEAM SEARCH PARAMETERS ==========
    beam_width: int = 5  # Number of partial placements kept at each depth
    beam_max_depth: int = 4  # Search depth per slot
    beam_candidates_per_step: int = 8  # Candidate moves expanded from each beam state
    beam_allow_remove: bool = True  # Allow remove moves during beam search

    # ========== LOOKAHEAD HINDSIGHT MILP ==========
    lookahead_window: int = 3  # Number of future slots to include in rolling-horizon MILP
    lookahead_discount: float = 1.0  # Optional discount across the lookahead horizon
    hindsight_r_nom_surrogate_weight: float = 0.0  # Linear surrogate weight for R_nom in lookahead MILP

    # ========== ADAPTIVE ENSEMBLE ==========
    ensemble_solver_types: List[str] = field(default_factory=lambda: ["beam_search", "greedy", "gnn_ppo"])
    ensemble_window: int = 5  # Rolling history window for solver selection
    ensemble_warmup_slots: int = 0  # Use current-slot score until this many slots are seen
    ensemble_use_counterfactual: bool = True  # Track all candidate scores vs selected-only
    ensemble_bandit_method: str = "sw_ucb"  # Options: "ucb", "thompson", "exp3", "sw_ucb"
    ensemble_exploration_c: float = 0.7  # UCB exploration strength
    ensemble_exp3_gamma: float = 0.1  # EXP3 exploration mixing weight
    ensemble_context_weight: float = 0.2  # Weight of contextual linear prediction in scoring
    ensemble_lin_ridge: float = 1.0  # Ridge regularization for contextual linear model

    # ========== GNN PPO SOLVER ==========
    gnn_ppo_embed_dim: int = 8
    gnn_ppo_max_steps: int = 16
    gnn_ppo_max_candidates: int = 200
    gnn_ppo_gamma: float = 0.98
    gnn_ppo_gae_lambda: float = 0.95
    gnn_ppo_clip_eps: float = 0.2
    gnn_ppo_entropy_coef: float = 0.01
    gnn_ppo_actor_lr: float = 0.01
    gnn_ppo_critic_lr: float = 0.02
    gnn_ppo_seed: int = 7
    gnn_ppo_temperature: float = 0.8
    gnn_ppo_explore_eps: float = 0.05
    gnn_ppo_topk_eval: int = 12
    gnn_ppo_min_improvement: float = 1e-5
    gnn_ppo_safe_mode: bool = True
    
    # ========== DATASET LOADING ==========
    dataset_source: str = "synthetic"  # Options: "synthetic", "netflix", "spotify", "custom"
    custom_dataset_path: Optional[str] = None  # Path to custom CSV
    
    # ========== PARALLEL PARAMETRIC EXPLORATION ==========
    parametric_lambda_remove_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.5, 1.0])
    num_parallel_workers: int = 4  # Number of CPU cores for parallel solving
    
    # ========== VISUALIZATION ==========
    plot_per_slot: bool = True  # Show detailed per-slot results
    plot_summary: bool = True  # Show aggregate plots
    save_plots: bool = False  # Save plots to disk
    plot_output_dir: str = "./results"  # Output directory
    verbose: bool = True  # Print solver/data progress to terminal
    
    def __post_init__(self):
        """Initialize default values for dictionaries"""
        if not self.dataset_sizes:
            self.dataset_sizes = {i: 10.0 + i*5 for i in range(self.num_datasets)}
        if not self.hop_budgets:
            self.hop_budgets = {i: 1 for i in range(self.num_datasets)}
        if not self.server_capacities:
            self.server_capacities = {j: 50.0 for j in range(self.num_servers)}
        if not self.server_bandwidth:
            self.server_bandwidth = {j: 30.0 for j in range(self.num_servers)}
        if not self.adjacency:
            self.adjacency = generate_random_connected_adjacency(num_servers=self.num_servers, edge_probability=0.5)
    
    def get_add_cost(self, i: int, j: int, t: int) -> float:
        """Cost of adding dataset i to server j at time t"""
        return self.dataset_sizes[i] * 0.1  # Simple linear model
    
    def override(self, **kwargs):
        """Override any configuration parameter"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Regenerate dependent dictionaries if num_datasets or num_servers changed
        if 'num_datasets' in kwargs:
            if not kwargs.get('dataset_sizes'):  # Only if not manually provided
                self.dataset_sizes = {i: 10.0 + i*5 for i in range(self.num_datasets)}
            if not kwargs.get('hop_budgets'):
                self.hop_budgets = {i: 2 + i % 2 for i in range(self.num_datasets)}
        
        if 'num_servers' in kwargs:
            if not kwargs.get('server_capacities'):
                self.server_capacities = {j: 50.0 for j in range(self.num_servers)}
            if not kwargs.get('server_bandwidth'):
                self.server_bandwidth = {j: 30.0 for j in range(self.num_servers)}
            if not kwargs.get('adjacency'):
                self.adjacency = generate_random_connected_adjacency(num_servers=self.num_servers, edge_probability=0.5)

    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'T': self.T,
            'num_datasets': self.num_datasets,
            'num_servers': self.num_servers,
            'rho': self.rho,
            'K_failures': self.K_failures,
            'Gamma_budget': self.Gamma_budget,
            'lambda_add': self.lambda_add,
            'solver_type': self.solver_type,
            'use_robust': self.use_robust
        }
    
    def print_summary(self):
        """Print configuration summary"""
        if not self.verbose:
            return
        print("="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        print(f"Time Horizon: {self.T} slots")
        print(f"Datasets: {self.num_datasets}, Servers: {self.num_servers}")
        print(f"Hop Budgets: {self.hop_budgets}")
        print(f"K-Failure Budget: {self.K_failures}")
        print(f"BS Uncertainty Budget (Γ): {self.Gamma_budget}")
        print(f"Blend Parameter (ρ): {self.rho}")
        print(f"Solver: {self.solver_type}")
        print(f"Robust Optimization: {'Enabled' if self.use_robust else 'Disabled'}")
        visualize_adjacency(self.adjacency, title='My Network')
        print("="*80)

# Create default configuration
# config = uEDDEConfig()
# print("✓ Configuration class defined")
# print("\nUsage:")
# print("  config = uEDDEConfig()")
# print("  config.override(T=10, rho=0.7, solver_type='greedy_adr')")