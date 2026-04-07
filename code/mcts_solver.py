import time
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from solver_base import BaseSolver, SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator

# MCTS node for dataset-specific tree
class DatasetMCTSNode:
    def __init__(self, dataset_state: Dict[int, int], server_idx: int = -1, parent=None, level: int = 0):
        """
        dataset_state: Dict[j -> 0/1] for dataset i across all servers
        server_idx: which server this node represents (-1 for root)
        level: tree depth (number of servers decided so far)
        """
        self.dataset_state = dataset_state  # Dict[server_j -> 0/1]
        self.server_idx = server_idx
        self.parent = parent
        self.level = level  # Track tree depth explicitly
        self.children: List["DatasetMCTSNode"] = []
        self.visits = 0
        self.value = 0.0

# MCTS solver - refactored for dataset-by-dataset approach
class MCTSSolver(BaseSolver):
    """
    Dataset-by-dataset MCTS solver.
    Each dataset gets its own MCTS tree where levels = servers.
    """

    def __init__(
        self,
        config: uEDDEConfig,
        data_gen: DataGenerator,
        iterations: int = 500,
        exploration_c: float = 1.1,
        seed: int = 42,
        unfulfilled_penalty: float = 100,
    ):
        super().__init__(config, data_gen)
        self.iterations = int(iterations)
        self.c = float(exploration_c)
        self.calculator = ObjectiveCalculator(config, data_gen)
        self.unfulfilled_penalty = unfulfilled_penalty
        random.seed(seed)
        np.random.seed(seed)

    # === Public API ===
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        start = time.time()
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        
        # Track final placement
        final_state = A_prev.copy()
        
        # Step 1: Identify datasets to optimize (request vector changed)
        datasets_to_optimize = self._identify_datasets_to_optimize(t)
        
        # Step 2: MCTS for each dataset
        for i in datasets_to_optimize:
            print(f"[MCTS] Optimizing dataset {i} at slot {t}")
            
            # Extract dataset state from A_prev (original baseline, not incremental final_state)
            # This keeps cost calculations consistent across all datasets
            current_dataset_state = {j: A_prev.get((i, j), 0) for j in J}
            
            # Run MCTS for this dataset
            optimized_state = self._mcts_for_dataset(
                i, current_dataset_state, t, A_prev, final_state
            )
            
            # Update final state with optimized placement
            for j in J:
                final_state[(i, j)] = optimized_state[j]
        
        # Step 3: Greedy fallback for unfulfillable requests
        for i in datasets_to_optimize:
            unfulfilled = self._find_unfulfilled_requests(i, t, final_state)
            if unfulfilled:
                print(f"[Greedy] Adding dataset {i} greedily for {len(unfulfilled)} attachment points")
                for p in unfulfilled:
                    # Find cheapest server with capacity
                    best_j = self._find_cheapest_server(i, None, p, t, final_state)
                    if best_j is not None:
                        final_state[(i, best_j)] = 1
        
        # Step 4: Build solution and calculate objective
        # final_state now includes both MCTS decisions and greedy additions
        sol = self._build_solution(final_state, t, A_prev)
        sol.solve_time = time.time() - start
        return sol

    # === Heuristics ===
    def _identify_datasets_to_optimize(self, t: int) -> List[int]:
        """
        Identify datasets with requests at time t.
        At t=0, optimize all active datasets (no previous requests to compare).
        At t>0, only optimize datasets that have requests at time t.
        """
        I = range(self.config.num_datasets)
        datasets_to_opt = []
        
        for i in I:
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            
            # At t=0, always optimize all active datasets
            if t == 0:
                datasets_to_opt.append(i)
                continue
            
            # At t>0, check if this dataset has ANY requests at time t
            has_requests = any(
                self.data.counts.get((i, p, t), 0) > 0
                for p in self.data.attachment_points.get(t, [])
            )
            
            if has_requests:
                datasets_to_opt.append(i)
            else:
                print(f"[Heuristic] Skipping dataset {i} (no requests at slot {t})")
        
        return datasets_to_opt

    def _get_server_ordering_heuristic(self, i: int, t: int) -> List[int]:
        """
        Order servers by hop distance to attachment points for dataset i.
        Servers closer to attachment points come first.
        """
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])
        server_scores = {}
        for j in J:
            score = 0
            for p in P_t:
                if self.data.counts.get((i, p, t), 0) > 0:
                    hop = self.config.hop_distances.get((j, p), float('inf'))
                    score += -hop  # negative so lower hops = higher score
            server_scores[j] = score
        
        # Sort by score (descending)
        ordered = sorted(J, key=lambda j: server_scores[j], reverse=True)
        return ordered

    # === Dataset-specific MCTS ===
    def _mcts_for_dataset(
        self,
        i: int,
        initial_state: Dict[int, int],
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        current_global_state: Dict[Tuple[int, int], int],
    ) -> Dict[int, int]:
        """
        Run MCTS for a single dataset across all servers.
        Returns optimized placement: Dict[server_j -> 0/1]
        """
        J = range(self.config.num_servers)
        server_order = self._get_server_ordering_heuristic(i, t)
        
        root = DatasetMCTSNode(initial_state.copy(), server_idx=-1, level=0)
        
        # MCTS iterations
        for _ in range(self.iterations):
            node = self._dataset_select(root, i, t, A_prev, current_global_state, server_order)
            
            # Evaluate at leaf
            reward = self._dataset_evaluate_leaf(
                i, node.dataset_state, t, A_prev, current_global_state, server_order
            )
            
            self._backpropagate(node, reward)
        
        # Choose best child (most visited)
        if root.children:
            best = max(root.children, key=lambda n: n.visits)
            final_state = best.dataset_state
        else:
            final_state = initial_state.copy()
        
        return final_state

    def _dataset_select(
        self,
        node: DatasetMCTSNode,
        i: int,
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        global_state: Dict[Tuple[int, int], int],
        server_order: List[int],
    ) -> DatasetMCTSNode:
        """
        Descend tree until we reach a node that needs expansion.
        Each level = one server. Track depth with node.level (not dict length).
        """
        while True:
            # Use explicit level tracking
            current_level = node.level
            
            if current_level >= len(server_order):
                # All servers visited (leaf reached)
                return node
            
            if not node.children:
                # Expand this node
                return self._dataset_expand(node, i, t, A_prev, global_state, server_order)
            
            # Select best child by UCB
            node = max(node.children, key=lambda c: self._ucb_dataset(node, c))

    def _dataset_expand(
        self,
        node: DatasetMCTSNode,
        i: int,
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        global_state: Dict[Tuple[int, int], int],
        server_order: List[int],
    ) -> DatasetMCTSNode:
        """
        Create 2 children: toggle dataset i on current server (add/remove/keep).
        Track tree depth with explicit level parameter.
        """
        current_level = node.level
        if current_level >= len(server_order):
            return node
        
        server_j = server_order[current_level]
        current_presence = node.dataset_state.get(server_j, 0)
        
        # Child 1: Toggle presence
        new_state_toggle = node.dataset_state.copy()
        new_state_toggle[server_j] = 1 - current_presence
        child_toggle = DatasetMCTSNode(new_state_toggle, server_idx=server_j, parent=node, level=current_level + 1)
        node.children.append(child_toggle)
        
        # Child 2: Keep same (only if different from toggle)
        if len(node.children) == 1:
            new_state_keep = node.dataset_state.copy()
            new_state_keep[server_j] = current_presence
            child_keep = DatasetMCTSNode(new_state_keep, server_idx=server_j, parent=node, level=current_level + 1)
            node.children.append(child_keep)
        
        return random.choice(node.children)

    def _dataset_evaluate_leaf(
        self,
        i: int,
        dataset_state: Dict[int, int],
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        global_state: Dict[Tuple[int, int], int],
        server_order: List[int],
    ) -> float:
        """
        Evaluate leaf node: single-dataset objective + unfulfilled penalty.
        """
        J = range(self.config.num_servers)
        
        # Check for unfulfilled requests
        unfulfilled = self._find_unfulfilled_requests_for_state(i, t, dataset_state, global_state)
        
        # Calculate single-dataset objective
        reward = self._calculate_single_dataset_objective(
            i, dataset_state, t, A_prev, global_state
        )
        
        # Add normalized unfulfilled penalty (scaled to match objective magnitude)
        # Normalize by total requests to avoid dominating the objective
        P_t = self.data.attachment_points.get(t, [])
        N_t = sum(self.data.counts.get((i, p, t), 0) for p in P_t)
        
        if N_t > 0 and len(unfulfilled) > 0:
            # Unfulfilled penalty per request (normalized)
            unfulfilled_penalty_normalized = self.unfulfilled_penalty * len(unfulfilled) / N_t
            reward -= unfulfilled_penalty_normalized
        
        return reward

    def _find_unfulfilled_requests(self, i: int, t: int, state: Dict[Tuple[int, int], int]) -> Set[int]:
        """
        Find attachment points where dataset i has requests but no nearby server has it.
        """
        P_t = self.data.attachment_points.get(t, [])
        H_i = self.config.hop_budgets[i]
        unfulfilled = set()
        
        for p in P_t:
            if self.data.counts.get((i, p, t), 0) == 0:
                continue
            
            # Check if any server within hop budget has dataset i
            found = False
            for j in range(self.config.num_servers):
                if state.get((i, j), 0) == 1:
                    hop = self.config.hop_distances.get((j, p), float('inf'))
                    if hop <= H_i:
                        found = True
                        break
            
            if not found:
                unfulfilled.add(p)
        
        return unfulfilled

    def _find_unfulfilled_requests_for_state(
        self, i: int, t: int, dataset_state: Dict[int, int], global_state: Dict[Tuple[int, int], int]
    ) -> Set[int]:
        """
        Check unfulfilled for single dataset (used during MCTS).
        """
        P_t = self.data.attachment_points.get(t, [])
        H_i = self.config.hop_budgets[i]
        J = range(self.config.num_servers)
        unfulfilled = set()
        
        for p in P_t:
            if self.data.counts.get((i, p, t), 0) == 0:
                continue
            
            found = False
            for j in J:
                # Check both dataset_state (for this dataset) and global_state (for others)
                presence = dataset_state.get(j, 0)
                hop = self.config.hop_distances.get((j, p), float('inf'))
                if presence == 1 and hop <= H_i:
                    found = True
                    break
            
            if not found:
                unfulfilled.add(p)
        
        return unfulfilled

    def _find_cheapest_server(
        self, i: int, j: int, p: int, t: int, state: Dict[Tuple[int, int], int]
    ) -> int:
        """
        Find cheapest server to add dataset i (greedy fallback).
        """
        J = range(self.config.num_servers)
        H_i = self.config.hop_budgets[i]
        
        best_j = None
        best_cost = float('inf')
        
        for j in J:
            if state.get((i, j), 0) == 1:
                continue  # Already has it
            
            hop = self.config.hop_distances.get((j, p), float('inf'))
            if hop > H_i:
                continue  # Out of hop budget
            
            # Check capacity
            used = sum(
                self.config.dataset_sizes[ii] * state.get((ii, j), 0)
                for ii in range(self.config.num_datasets)
            )
            if used + self.config.dataset_sizes[i] > self.config.server_capacities[j]:
                continue  # No capacity
            
            cost = self.config.get_add_cost(i, j, t)
            if cost < best_cost:
                best_cost = cost
                best_j = j
        
        return best_j

    # === Single-dataset objective ===
    def _calculate_single_dataset_objective(
        self,
        i: int,
        dataset_state: Dict[int, int],
        t: int,
        A_prev: Dict[Tuple[int, int], int],
        global_state: Dict[Tuple[int, int], int],
    ) -> float:
        """
        Calculate objective considering only dataset i.
        Assumes all other datasets stay in global_state.
        """
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])
        K = int(self.config.K_failures)
        H_i = self.config.hop_budgets[i]
        
        # R_nom: removal fraction for dataset i
        prev_size = sum(self.config.dataset_sizes[i] for j in J if A_prev.get((i, j), 0) == 1)
        removed_size = sum(self.config.dataset_sizes[i] for j in J if A_prev.get((i, j), 0) == 1 and dataset_state.get(j, 0) == 0)
        R_nom = removed_size / prev_size if prev_size > 0 else 0.0
        
        # B_nom: ring benefit nominal for dataset i only
        N_i = sum(self.data.counts.get((i, p, t), 0) for p in P_t)
        B_sum = 0.0
        
        if N_i > 0:
            for p in P_t:
                n_ip = self.data.counts.get((i, p, t), 0)
                if n_ip <= 0:
                    continue
                
                hops = [self.config.hop_distances.get((j, p), float('inf')) for j in J if dataset_state.get(j, 0) == 1]
                if not hops:
                    continue
                
                best = min(hops)
                if best <= H_i:
                    w_bar = self.data.weights_nominal.get((i, p, t), 1.0)
                    B_sum += w_bar * n_ip * max(0.0, H_i - best)
        
        B_nom = B_sum / (H_i * N_i) if H_i > 0 and N_i > 0 else 0.0
        
        # B_wc: worst-case benefit for dataset i
        B_wc = 0.0
        if N_i > 0:
            total_wc = 0.0
            for p in P_t:
                n_ip = self.data.counts.get((i, p, t), 0)
                if n_ip <= 0:
                    continue
                
                hops = sorted(self.config.hop_distances.get((j, p), float('inf')) for j in J if dataset_state.get(j, 0) == 1)
                wc_hop = hops[K] if len(hops) > K else H_i
                
                if wc_hop <= H_i:
                    w_bar = self.data.weights_nominal.get((i, p, t), 1.0)
                    total_wc += w_bar * n_ip * max(0.0, H_i - wc_hop)
            
            B_wc = total_wc / (H_i * N_i) if H_i > 0 and N_i > 0 else 0.0
        
        # Op cost: adds for dataset i only
        Op = self.config.lambda_add * sum(
            self.config.get_add_cost(i, j, t)
            for j in J
            if dataset_state.get(j, 0) == 1 and A_prev.get((i, j), 0) == 0
        )
        
        # Combine objectives
        if self.config.use_robust:
            objective = R_nom + B_wc - Op
        else:
            objective = R_nom + B_nom - Op
        
        # Return objective directly (higher is better)
        # Do NOT negate - we want higher rewards for better solutions
        return objective

    def _backpropagate(self, node: DatasetMCTSNode, reward: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    def _ucb_dataset(self, parent: DatasetMCTSNode, child: DatasetMCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + self.c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)

    # === Build Solution ===
    def _build_solution(
        self,
        final_state: Dict[Tuple[int, int], int],
        t: int,
        A_prev: Dict[Tuple[int, int], int],
    ) -> SlotSolution:
        sol = SlotSolution(time_slot=t, status="Feasible")
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        
        # Track states, adds, removes
        for i in I:
            for j in J:
                prev = A_prev.get((i, j), 0)
                curr = final_state.get((i, j), 0)
                
                if curr == 1:
                    sol.states[(i, j)] = 1
                
                if prev == 0 and curr == 1:
                    sol.adds[(i, j)] = 1
                
                if prev == 1 and curr == 0:
                    sol.removes[(i, j)] = 1
        
        # Calculate full objective with all datasets
        # ObjectiveCalculator compares final_state vs A_prev and handles all costs
    
        breakdown = self.calculator.calculate(
            t=t,
            state_current=final_state,
            state_prev=A_prev
        )
        
        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.R_wc = float(breakdown.R_wc)
        sol.B_wc = float(breakdown.B_wc)
        sol.Op_cost = float(breakdown.Op_cost)
        sol.objective_value = float(breakdown.objective_total if self.config.use_robust else breakdown.objective_nominal)
        
        return sol