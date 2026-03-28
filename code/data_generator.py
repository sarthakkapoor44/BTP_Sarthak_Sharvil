from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import networkx as nx
from config import uEDDEConfig
import numpy as np

class DataGenerator:
    """
    Generate or load data for uEDDE problem
    Supports: synthetic, Netflix, Spotify, or custom datasets
    """
    
    def __init__(self, config: uEDDEConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Compute hop distances from adjacency
        self._compute_hop_distances()
        
        # Data structures (to be populated)
        self.initial_state: Dict[Tuple[int, int], int] = {}  # A_{ij,0}
        self.counts: Dict[Tuple[int, int, int], int] = {}  # N^i_{p,t}
        self.weights_nominal: Dict[Tuple[int, int, int], float] = {}  # bar_Upsilon_{ip,t}
        self.weights_error: Dict[Tuple[int, int, int], float] = {}  # hat_Upsilon_{ip,t}
        self.active_datasets: Dict[Tuple[int, int], int] = {}  # alpha_{i,t}
        self.attachment_points: Dict[int, List[int]] = {}  # P_t
        
    def _compute_hop_distances(self):
        """Compute all-pairs shortest paths (hop distances)"""
        if self.config.hop_distances:
            print("✓ Using precomputed hop distances")
            return
        
        G = nx.Graph()
        for node, neighbors in self.config.adjacency.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        self.config.hop_distances = {}
        for source in range(self.config.num_servers):
            try:
                lengths = nx.single_source_shortest_path_length(G, source)
            except:
                lengths = {source: 0}
            
            for target in range(self.config.num_servers):
                self.config.hop_distances[(source, target)] = lengths.get(target, float('inf'))
        
        print(f"✓ Computed hop distances for {self.config.num_servers} servers")
    
    def get_neighborhood(self, p: int, hop_budget: int) -> List[int]:
        """Get neighborhood N_p(H_i) = {j: h_{j,p} <= H_i}"""
        neighborhood = []
        for j in range(self.config.num_servers):
            if self.config.hop_distances.get((j, p), float('inf')) <= hop_budget:
                neighborhood.append(j)
        return neighborhood
    
    # ========== INITIAL STATE ==========
    
    def generate_initial_state(self, manual_state: Optional[Dict] = None, 
                               sparsity: float = 0.3):
        """Generate or load initial placement A_{ij,0}"""
        if manual_state is not None:
            self.initial_state = manual_state
            print("✓ Using manual initial state")
        else:
            self.initial_state = {}
            for i in range(self.config.num_datasets):
                for j in range(self.config.num_servers):
                    self.initial_state[(i, j)] = 1 if self.rng.random() < sparsity else 0
            print(f"✓ Generated random initial state (sparsity={sparsity})")
        
        # Validate capacity
        for j in range(self.config.num_servers):
            total = sum(self.config.dataset_sizes[i] * self.initial_state[(i, j)] 
                       for i in range(self.config.num_datasets))
            if total > self.config.server_capacities[j]:
                print(f"⚠ Warning: Server {j} capacity violated ({total:.1f} > {self.config.server_capacities[j]})")
        
        return self.initial_state
    
    # ========== REQUEST COUNTS ==========
    
    def generate_counts(self, manual_counts: Optional[Dict] = None):
        """Generate request counts N^i_{p,t}"""
        if manual_counts is not None:
            self.counts = manual_counts
            self._extract_attachment_points()
            print("✓ Using manual request counts")
        else:
            self._generate_synthetic_counts()
            print(f"✓ Generated synthetic request counts for {self.config.T} time slots")
        return self.counts
    
    def _generate_synthetic_counts(self):
        """Generate synthetic Poisson-like counts"""
        self.counts = {}
        self.attachment_points = {}
        
        for t in range(1, self.config.T + 1):
            # Random active APs
            num_active = self.rng.integers(2, self.config.num_servers + 1)
            active_aps = self.rng.choice(self.config.num_servers, num_active, replace=False)
            self.attachment_points[t] = list(active_aps)
            
            for p in active_aps:
                for i in range(self.config.num_datasets):
                    # Poisson arrivals with time-varying rate
                    base_rate = 5 + 2 * i
                    time_factor = 1 + 0.3 * np.sin(2 * np.pi * t / self.config.T)
                    count = self.rng.poisson(base_rate * time_factor)
                    self.counts[(i, p, t)] = count
    
    def _extract_attachment_points(self):
        """Extract P_t from manual counts"""
        self.attachment_points = {}
        for (i, p, t) in self.counts.keys():
            if t not in self.attachment_points:
                self.attachment_points[t] = []
            if p not in self.attachment_points[t]:
                self.attachment_points[t].append(p)
        # print("Attachment points for every time:")
        # print(self.attachment_points[1435])
    
    def _format_compact_list(self, values, max_items: int = 12) -> str:
        values = list(values)
        if len(values) <= max_items:
            return str(values)
        head = values[: max_items // 2]
        tail = values[-(max_items // 2):]
        return f"{head} ... {tail} (total={len(values)})"

    def load_dataset(self, file_path: str):
        """Load dataset from CSV with columns: timestamp, server, dataset"""
        if self.config.verbose:
            print(f"Loading dataset from {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'server', 'dataset']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}, found: {list(df.columns)}")
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Map timestamps to time slots (1 to T)
            # Ensure timestamps are mapped uniformly across the range
            unique_timestamps = sorted(df['timestamp'].unique())
            if len(unique_timestamps) <= self.config.T:
                # Few unique timestamps: map directly
                timestamp_to_slot = {ts: min(idx + 1, self.config.T) 
                                    for idx, ts in enumerate(unique_timestamps)}
            else:
                # Many timestamps: bin them into T slots
                df['time_slot'] = pd.cut(df['timestamp'], bins=self.config.T, labels=False) + 1
            
            # If we used direct mapping, apply it
            if 'time_slot' not in df.columns:
                df['time_slot'] = df['timestamp'].map(timestamp_to_slot)
            
            # Aggregate counts: group by (dataset, server, time_slot) and count occurrences
            self.counts = {}
            grouped = df.groupby(['dataset', 'server', 'time_slot']).size()
            # print(grouped.items())
            for (dataset_id, server_id, time_slot), count in grouped.items():
                # Validate indices
                if dataset_id >= self.config.num_datasets:
                    continue  # Skip datasets beyond configured range
                if server_id >= self.config.num_servers:
                    continue  # Skip servers beyond configured range
                if time_slot < 1 or time_slot > self.config.T:
                    continue  # Skip invalid time slots
                
                # Store count: (dataset_id, server_id, time_slot) -> count
                key = (int(dataset_id), int(server_id), int(time_slot))
                self.counts[key] = int(count)
            # print(self.counts)
            
            # Extract attachment points (servers that have requests in each time slot)
            self._extract_attachment_points()
            
            if self.config.verbose:
                time_slots = sorted(set(t for _, _, t in self.counts.keys()))
                datasets = sorted(set(i for i, _, _ in self.counts.keys()))
                aps = sorted(set(p for _, p, _ in self.counts.keys()))
                print(f"✓ Loaded {len(self.counts)} request records from dataset")
                print(f"  Time slots: {self._format_compact_list(time_slots)}")
                print(f"  Datasets: {self._format_compact_list(datasets)}")
                print(f"  Servers (APs): {self._format_compact_list(aps)}")
            
        except Exception as e:
            if self.config.verbose:
                print(f"✗ Error loading dataset: {e}")
                print("  Falling back to synthetic data")
            self._generate_synthetic_counts()
    
    # ========== ACTIVE DATASETS ==========
    
    def generate_active_datasets(self, manual_active: Optional[Dict] = None,
                                 activity_prob: float = 0.8):
        """Generate active dataset indicators alpha_{i,t}"""
        if manual_active is not None:
            self.active_datasets = manual_active
            print("✓ Using manual active datasets")
        else:
            self.active_datasets = {}
            for t in range(1, self.config.T + 1):
                for i in range(self.config.num_datasets):
                    self.active_datasets[(i, t)] = 1 if self.rng.random() < activity_prob else 0
            
            # Enforce C0-count: inactive datasets have zero counts
            for (i, p, t), count in list(self.counts.items()):
                if self.active_datasets.get((i, t), 0) == 0:
                    self.counts[(i, p, t)] = 0
            
            print(f"✓ Generated active dataset indicators (activity={activity_prob})")
        
        return self.active_datasets
    
    # ========== WEIGHTS (UNCERTAINTY) ==========
    
    def generate_weights(self, manual_nominal: Optional[Dict] = None,
                        manual_error: Optional[Dict] = None,
                        error_pct_range: Tuple[float, float] = (0.1, 0.3)):
        """Generate nominal and error weights for BS uncertainty"""
        if manual_nominal is not None and manual_error is not None:
            self.weights_nominal = manual_nominal
            self.weights_error = manual_error
            print("✓ Using manual weights")
        else:
            self.weights_nominal = {}
            self.weights_error = {}
            
            for t in range(1, self.config.T + 1):
                for p in self.attachment_points.get(t, []):
                    for i in range(self.config.num_datasets):
                        # Nominal weight (uniform around 1.0)
                        self.weights_nominal[(i, p, t)] = 0.8 + 0.4 * self.rng.random()
                        
                        # Error bound (percentage of nominal)
                        error_pct = error_pct_range[0] + (error_pct_range[1] - error_pct_range[0]) * self.rng.random()
                        self.weights_error[(i, p, t)] = self.weights_nominal[(i, p, t)] * error_pct
            
            print(f"✓ Generated weights with error range {error_pct_range}")
        
        return self.weights_nominal, self.weights_error
    
    # ========== MASTER GENERATION ==========
    
    def generate_all(self, dataset_source: Optional[str] = None,activity = 0.8):
        """Generate all data based on configuration"""
        source = dataset_source or self.config.dataset_source
        
        self.generate_initial_state()
        
        if source == "data" and self.config.custom_dataset_path:
            self.load_dataset(self.config.custom_dataset_path)
        elif source == "synthetic":
            self.generate_counts()
        else:
            print(f"⚠ Unknown dataset source: {source}, using synthetic")
            self.generate_counts()
        self.generate_active_datasets(activity_prob=activity)
        self.generate_weights()
        
        return self
    
    def print_summary(self):
        """Print data summary"""
        if not self.config.verbose:
            return
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(f"Initial placements: {sum(self.initial_state.values())} replicas")
        print(f"Total requests: {sum(self.counts.values())}")
        slot_keys = sorted(self.attachment_points.keys())
        print(f"Time slots with requests: {self._format_compact_list(slot_keys)}")
        print(f"Unique attachment points: {len(set(p for t in self.attachment_points.values() for p in t))}")
        print(f"Active datasets (avg): {np.mean([sum(self.active_datasets.get((i,t),0) for i in range(self.config.num_datasets)) for t in range(1, self.config.T+1)]):.1f}")
        print("="*80)

