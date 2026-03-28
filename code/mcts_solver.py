import time
import math
import random
import numpy as np
from typing import List, Dict, Tuple
from solver_base import BaseSolver,SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from objective_calculator import ObjectiveCalculator

# MCTS node
class MCTSNode:
    def __init__(self, state: Dict[Tuple[int,int], int], parent=None, action=None):
        self.state = state                  # Dict[(i,j)->0/1]
        self.parent = parent
        self.action = action                # ("add"/"remove", i, j)
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0                    # accumulated reward

# MCTS solver
class MCTSSolver(BaseSolver):
    """
    Monte Carlo Tree Search solver for single-slot first-stage decisions.
    Keeps evaluation logic self-contained (R_nom, B_nom, B_wc, Op).
    """

    def __init__(
        self,
        config: uEDDEConfig,
        data_gen: DataGenerator,
        iterations: int = 500,
        rollout_depth: int = 4,
        exploration_c: float = 1.4,
        seed: int = 42,
    ):
        super().__init__(config, data_gen)
        self.iterations = int(iterations)
        self.rollout_depth = int(rollout_depth)
        self.c = float(exploration_c)
        self.calculator = ObjectiveCalculator(config, data_gen)
        random.seed(seed)
        np.random.seed(seed)

    # === public API ===
    def solve_slot(self, t: int, A_prev: Dict[Tuple[int,int], int]) -> SlotSolution:
        start = time.time()
        root = MCTSNode(A_prev.copy())

        # Iterative MCTS
        for _ in range(self.iterations):
            node = self._select(root, t)
            reward = self._rollout(node.state, t, A_prev)
            self._backpropagate(node, reward)

        # Choose best child (most visited) or fallback to A_prev
        if root.children:
            best = max(root.children, key=lambda n: n.visits)
            final_state = best.state
        else:
            final_state = A_prev.copy()

        sol = self._build_solution(final_state, t, A_prev)
        sol.solve_time = time.time() - start
        return sol

    # === MCTS core ===
    def _select(self, node: MCTSNode, t: int) -> MCTSNode:
        # descend until leaf or expandable node
        while True:
            if not node.children:
                return self._expand(node, t)
            node = max(node.children, key=lambda c: self._ucb(node, c))

    def _expand(self, node: MCTSNode, t: int) -> MCTSNode:
        actions = self._valid_actions(node.state, t)
        if not actions:
            return node
        # create children for all actions (could progressive-widen later)
        for act in actions:
            new_state = self._apply_action(node.state, act)
            child = MCTSNode(new_state, parent=node, action=act)
            node.children.append(child)
        return random.choice(node.children)

    def _rollout(self, state: Dict[Tuple[int,int],int], t: int, A_prev: Dict[Tuple[int,int],int]) -> float:
        sim_state = state.copy()
        for _ in range(self.rollout_depth):
            actions = self._valid_actions(sim_state, t)
            if not actions:
                break
            act = random.choice(actions)
            sim_state = self._apply_action(sim_state, act)
        return self._evaluate_state(sim_state, t, A_prev)

    def _backpropagate(self, node: MCTSNode, reward: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    def _ucb(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + self.c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)

    # === actions / state transition ===
    def _valid_actions(self, state: Dict[Tuple[int,int],int], t: int) -> List[Tuple[str,int,int]]:
        actions: List[Tuple[str,int,int]] = []
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)

        for i in I:
            # skip inactive datasets for this slot
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue
            for j in J:
                if state.get((i, j), 0) == 0:
                    # ADD candidate if capacity allows
                    used = sum(self.config.dataset_sizes[ii] * state.get((ii, j), 0) for ii in I)
                    if used + self.config.dataset_sizes[i] <= self.config.server_capacities[j]:
                        actions.append(("add", i, j))
                else:
                    # always allow remove (greedy may prune later)
                    actions.append(("remove", i, j))
        return actions

    def _apply_action(self, state: Dict[Tuple[int,int],int], action: Tuple[str,int,int]) -> Dict[Tuple[int,int],int]:
        new_state = state.copy()
        kind, i, j = action
        new_state[(i, j)] = 1 if kind == "add" else 0
        return new_state

    # === evaluation functions (self-contained) ===
    def _evaluate_state(self, state: Dict[Tuple[int,int],int], t: int, A_prev: Dict[Tuple[int,int],int]) -> float:
        breakdown = self.calculator.calculate(
            t=t,
            state_current=state,
            state_prev=A_prev
        )
        return (breakdown.objective_total if self.config.use_robust 
                else breakdown.objective_nominal)

    def _compute_nominal(self, state: Dict[Tuple[int,int],int], t: int, A_prev: Dict[Tuple[int,int],int]) -> Tuple[float,float,float]:
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])

        # R_nom (dedup removed fraction) — same denom as other solvers
        denom = sum(self.config.dataset_sizes[i] * A_prev.get((i, j), 0)
                    for i in I for j in J
                    if self.data.active_datasets.get((i, t), 0) == 1)
        removed_size = sum(self.config.dataset_sizes[i]
                           for i in I for j in J
                           if A_prev.get((i, j), 0) == 1 and state.get((i, j), 0) == 0)
        R_nom = removed_size / denom if denom > 0 else 0.0

        # B_nom (ring benefit, normalized)
        H_sum = sum(self.config.hop_budgets[i] for i in I if self.data.active_datasets.get((i,t),0)==1)
        N_sum = sum(self.data.counts.get((i,p,t),0) for i in I for p in P_t)
        B_sum = 0.0
        if H_sum > 0 and N_sum > 0:
            for i in I:
                if self.data.active_datasets.get((i, t), 0) != 1:
                    continue
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    n_ip = self.data.counts.get((i, p, t), 0)
                    if n_ip <= 0:
                        continue
                    # best hop among servers storing i in state
                    hops = [self.config.hop_distances.get((j, p), float('inf')) for j in J if state.get((i, j), 0) == 1]
                    if not hops:
                        continue
                    best = min(hops)
                    if best <= H_i:
                        w_bar = self.data.weights_nominal.get((i, p, t), 1.0)
                        B_sum += w_bar * n_ip * max(0.0, H_i - best)
        B_nom = B_sum / (H_sum * N_sum) if H_sum > 0 and N_sum > 0 else 0.0

        # Op cost (adds only, scaled by lambda_add)
        Op = self.config.lambda_add * sum(self.config.get_add_cost(i, j, t)
                                          for i in I for j in J
                                          if state.get((i, j), 0) == 1 and A_prev.get((i, j), 0) == 0)
        return R_nom, B_nom, Op

    def _compute_B_wc(self, state: Dict[Tuple[int,int],int], t: int) -> float:
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)
        P_t = self.data.attachment_points.get(t, [])
        K = int(self.config.K_failures)

        H_sum = sum(self.config.hop_budgets[i] for i in I if self.data.active_datasets.get((i,t),0)==1)
        N_sum = sum(self.data.counts.get((i,p,t),0) for i in I for p in P_t)

        total = 0.0
        if H_sum > 0 and N_sum > 0:
            for i in I:
                if self.data.active_datasets.get((i, t), 0) != 1:
                    continue
                H_i = self.config.hop_budgets[i]
                for p in P_t:
                    n_ip = self.data.counts.get((i, p, t), 0)
                    if n_ip <= 0:
                        continue
                    hops = sorted(self.config.hop_distances.get((j, p), float('inf'))
                                  for j in J if state.get((i, j), 0) == 1)
                    wc_hop = hops[K] if len(hops) > K else H_i
                    if wc_hop <= H_i:
                        w_bar = self.data.weights_nominal.get((i, p, t), 1.0)
                        total += w_bar * n_ip * max(0.0, H_i - wc_hop)
        return total / (H_sum * N_sum) if H_sum > 0 and N_sum > 0 else 0.0

    # === build SlotSolution object consistent with your framework ===
    def _build_solution(self, final_state: Dict[Tuple[int,int],int], t: int, A_prev: Dict[Tuple[int,int],int]) -> SlotSolution:
        sol = SlotSolution(time_slot=t, status="Feasible")
        I = range(self.config.num_datasets)
        J = range(self.config.num_servers)

        # states/adds/removes
        for i in I:
            for j in J:
                prev = A_prev.get((i,j), 0)
                curr = final_state.get((i,j), 0)
                if curr == 1:
                    sol.states[(i,j)] = 1
                if prev == 0 and curr == 1:
                    sol.adds[(i,j)] = 1
                if prev == 1 and curr == 0:
                    sol.removes[(i,j)] = 1

        # Use centralized calculator for all objectives
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
        
        sol.objective_value = (breakdown.objective_total if self.config.use_robust 
                              else breakdown.objective_nominal)
        return sol
