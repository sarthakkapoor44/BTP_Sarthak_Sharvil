from typing import Dict, Tuple, List
import time
import copy
import math

import numpy as np

from solver_base import BaseSolver, SlotSolution
from config import uEDDEConfig
from data_generator import DataGenerator
from beam_search_solver import BeamSearchSolver
from greedy_custom_solver import GreedyCustomSolver
from lyapunov_solver import LyapunovSolver
from robust_evaluator import RobustEvaluator, RobustEvaluation


class AdaptiveEnsembleSolver(BaseSolver):
    """
    No-peek contextual bandit selector over multiple candidate solvers.

    Decision at slot t uses only past observations and current context features,
    then executes only the chosen solver (unless counterfactual updates are enabled).

    Default candidate set: beam_search, greedy, gnn_ppo.
    """

    _SUPPORTED = {
        "beam_search": BeamSearchSolver,
        "greedy": GreedyCustomSolver,
        "lyapunov": LyapunovSolver,
        # "gnn_ppo": GNNPPOSolver,
    }

    def __init__(self, config: uEDDEConfig, data_gen: DataGenerator):
        super().__init__(config, data_gen)
        requested = list(getattr(config, "ensemble_solver_types", ["beam_search", "greedy"]))
        self.window = max(1, int(getattr(config, "ensemble_window", 5)))
        self.warmup_slots = max(0, int(getattr(config, "ensemble_warmup_slots", 0)))
        self.use_counterfactual = bool(getattr(config, "ensemble_use_counterfactual", True))
        self.bandit_method = str(getattr(config, "ensemble_bandit_method", "sw_ucb")).lower().strip()
        self.exploration_c = float(getattr(config, "ensemble_exploration_c", 0.7))
        self.exp3_gamma = float(getattr(config, "ensemble_exp3_gamma", 0.1))
        self.context_weight = float(getattr(config, "ensemble_context_weight", 0.2))
        self.lin_ridge = float(getattr(config, "ensemble_lin_ridge", 1.0))

        self._allowed_methods = {"ucb", "thompson", "exp3", "sw_ucb"}
        if self.bandit_method not in self._allowed_methods:
            raise ValueError(
                f"Unsupported ensemble_bandit_method '{self.bandit_method}'. "
                f"Supported: {sorted(self._allowed_methods)}"
            )

        if not requested:
            requested = ["beam_search", "greedy", "gnn_ppo"]

        self.candidate_names: List[str] = []
        self.candidate_solvers: Dict[str, BaseSolver] = {}
        for raw_name in requested:
            name = str(raw_name).lower().strip()
            if name not in self._SUPPORTED:
                raise ValueError(
                    f"Unsupported ensemble solver '{raw_name}'. "
                    f"Supported: {sorted(self._SUPPORTED.keys())}"
                )
            if name in self.candidate_solvers:
                continue
            self.candidate_names.append(name)
            self.candidate_solvers[name] = self._SUPPORTED[name](config, data_gen)

        self.history: Dict[str, List[float]] = {name: [] for name in self.candidate_names}
        self.pull_counts: Dict[str, int] = {name: 0 for name in self.candidate_names}

        # Reward normalization state for robust online updates.
        self._reward_min: float = float("inf")
        self._reward_max: float = float("-inf")

        # Contextual linear model per arm (ridge-regularized online least squares).
        self.context_dim = 6  # [bias, entropy, active_ratio, uncovered_ratio, migration_pressure, t_norm]
        self.lin_A: Dict[str, np.ndarray] = {}
        self.lin_b: Dict[str, np.ndarray] = {}
        for name in self.candidate_names:
            self.lin_A[name] = self.lin_ridge * np.eye(self.context_dim, dtype=float)
            self.lin_b[name] = np.zeros(self.context_dim, dtype=float)

        # Thompson Beta params and EXP3 weights.
        self.ts_alpha: Dict[str, float] = {name: 1.0 for name in self.candidate_names}
        self.ts_beta: Dict[str, float] = {name: 1.0 for name in self.candidate_names}
        self.exp3_w: Dict[str, float] = {name: 1.0 for name in self.candidate_names}
        self._last_exp3_probs: Dict[str, float] = {name: 1.0 / len(self.candidate_names) for name in self.candidate_names}
        self.total_decisions: int = 0

        # Shared robust scorer for meta-level arm comparison.
        self.robust_evaluator = RobustEvaluator(config, data_gen)

    def _rolling_mean(self, values: List[float]) -> float:
        if not values:
            return float("-inf")
        subset = values[-self.window:]
        return float(sum(subset) / len(subset))

    def _context_features(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> np.ndarray:
        active = [i for i in range(self.config.num_datasets) if self.data.active_datasets.get((i, t), 0) == 1]
        active_count = len(active)
        active_ratio = (active_count / float(max(1, self.config.num_datasets)))

        demand_by_dataset = []
        total_dem = 0.0
        for i in range(self.config.num_datasets):
            val = 0.0
            for p in self.data.attachment_points.get(t, []):
                val += float(self.data.counts.get((i, p, t), 0))
            demand_by_dataset.append(val)
            total_dem += val

        if total_dem <= 0:
            demand_entropy = 0.0
        else:
            probs = [v / total_dem for v in demand_by_dataset if v > 0]
            ent = -sum(p * math.log(max(1e-12, p)) for p in probs)
            max_ent = math.log(max(2, self.config.num_datasets))
            demand_entropy = float(ent / max_ent) if max_ent > 0 else 0.0

        demand_pairs = 0
        uncovered_pairs = 0
        for i in active:
            H_i = self.config.hop_budgets[i]
            for p in self.data.attachment_points.get(t, []):
                req = int(self.data.counts.get((i, p, t), 0))
                if req <= 0:
                    continue
                demand_pairs += 1
                covered = any(
                    A_prev.get((i, j), 0) == 1 and self.config.hop_distances.get((j, p), float("inf")) <= H_i
                    for j in range(self.config.num_servers)
                )
                if not covered:
                    uncovered_pairs += 1
        uncovered_ratio = float(uncovered_pairs / max(1, demand_pairs))

        active_no_rep = 0
        for i in active:
            if all(A_prev.get((i, j), 0) == 0 for j in range(self.config.num_servers)):
                active_no_rep += 1
        migration_pressure = float(active_no_rep / max(1, active_count))

        t_norm = float((t - 1) / max(1, self.config.T - 1))

        return np.array([
            1.0,
            demand_entropy,
            active_ratio,
            uncovered_ratio,
            migration_pressure,
            t_norm,
        ], dtype=float)

    def _predict_context_reward(self, name: str, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.lin_A[name])
        theta = A_inv @ self.lin_b[name]
        return float(theta @ x)

    def _normalize_reward(self, raw_reward: float) -> float:
        self._reward_min = min(self._reward_min, raw_reward)
        self._reward_max = max(self._reward_max, raw_reward)
        span = self._reward_max - self._reward_min
        if span <= 1e-12:
            return 0.5
        val = (raw_reward - self._reward_min) / span
        return float(max(0.0, min(1.0, val)))

    def _choose_solver_no_peek(self, t: int, x: np.ndarray) -> str:
        if t <= self.warmup_slots:
            return self.candidate_names[(t - 1) % len(self.candidate_names)]

        self.total_decisions += 1
        n_arms = len(self.candidate_names)

        if self.bandit_method == "exp3":
            weight_sum = sum(self.exp3_w.values())
            probs: Dict[str, float] = {}
            for name in self.candidate_names:
                base = self.exp3_w[name] / max(1e-12, weight_sum)
                probs[name] = (1.0 - self.exp3_gamma) * base + (self.exp3_gamma / n_arms)
            self._last_exp3_probs = probs

            draw = np.random.random()
            cdf = 0.0
            for name in self.candidate_names:
                cdf += probs[name]
                if draw <= cdf:
                    return name
            return self.candidate_names[-1]

        # Deterministic argmax scorers for UCB-family and Thompson.
        scores: Dict[str, float] = {}
        for name in self.candidate_names:
            pred = self._predict_context_reward(name, x)

            if self.bandit_method == "thompson":
                sample = float(np.random.beta(self.ts_alpha[name], self.ts_beta[name]))
                scores[name] = sample + self.context_weight * pred
                continue

            if self.bandit_method == "sw_ucb":
                hist = self.history[name][-self.window:]
                n = len(hist)
                mean = float(sum(hist) / max(1, n)) if n > 0 else 0.0
                bonus = self.exploration_c * math.sqrt(math.log(self.total_decisions + 1.0) / max(1.0, n))
                scores[name] = mean + bonus + self.context_weight * pred
                continue

            # default ucb over full history
            n = self.pull_counts[name]
            mean = float(sum(self.history[name]) / max(1, len(self.history[name]))) if self.history[name] else 0.0
            bonus = self.exploration_c * math.sqrt(math.log(self.total_decisions + 1.0) / max(1.0, n))
            scores[name] = mean + bonus + self.context_weight * pred

        return max(self.candidate_names, key=lambda nm: scores[nm])

    def _update_one_arm(self, name: str, x: np.ndarray, raw_reward: float) -> None:
        norm_reward = self._normalize_reward(raw_reward)
        self.history[name].append(norm_reward)
        self.pull_counts[name] += 1

        # Context model update.
        self.lin_A[name] += np.outer(x, x)
        self.lin_b[name] += norm_reward * x

        # Thompson Beta update.
        self.ts_alpha[name] += norm_reward
        self.ts_beta[name] += (1.0 - norm_reward)

    def _update_after_decision(
        self,
        selected_name: str,
        context: np.ndarray,
        selected_obj: float,
        A_prev: Dict[Tuple[int, int], int],
        t: int,
    ) -> None:
        if self.bandit_method == "exp3":
            r_sel = self._normalize_reward(selected_obj)
            p_sel = self._last_exp3_probs.get(selected_name, 1.0 / max(1, len(self.candidate_names)))
            estimated_gain = r_sel / max(1e-9, p_sel)
            n_arms = len(self.candidate_names)
            self.exp3_w[selected_name] *= math.exp((self.exp3_gamma * estimated_gain) / max(1, n_arms))

        # Always update chosen arm with observed reward.
        self._update_one_arm(selected_name, context, selected_obj)

        # Optional counterfactual updates to improve learning under non-stationarity.
        if self.use_counterfactual:
            for name in self.candidate_names:
                if name == selected_name:
                    continue
                solver = self.candidate_solvers[name]
                candidate = solver.solve_slot(t, copy.deepcopy(A_prev))
                cand_eval = self.robust_evaluator.evaluate_placement(
                    t=t,
                    state_current=candidate.states,
                    state_prev=A_prev,
                )
                self._update_one_arm(name, context, float(cand_eval.objective_blended))

    def _apply_eval_to_solution(
        self,
        sol: SlotSolution,
        evaluation: RobustEvaluation,
        A_prev: Dict[Tuple[int, int], int],
    ) -> None:
        executed_state = dict(evaluation.post_recourse_state or sol.states)

        executed_adds: Dict[Tuple[int, int], int] = {}
        executed_removes: Dict[Tuple[int, int], int] = {}
        for i in range(self.config.num_datasets):
            for j in range(self.config.num_servers):
                prev = 1 if A_prev.get((i, j), 0) else 0
                now = 1 if executed_state.get((i, j), 0) else 0
                if prev == 0 and now == 1:
                    executed_adds[(i, j)] = 1
                elif prev == 1 and now == 0:
                    executed_removes[(i, j)] = 1

        sol.adds = executed_adds
        sol.removes = executed_removes
        sol.states = executed_state
        sol.R_nominal = float(evaluation.R_nominal)
        sol.B_nominal = float(evaluation.B_nominal)
        sol.Op_cost = float(evaluation.Op_cost)
        sol.R_wc = float(evaluation.R_wc)
        sol.B_wc = float(evaluation.B_wc)
        sol.objective_value = float(evaluation.objective_blended)
        sol.failures = dict(evaluation.worst_failures)
        sol.recourse_removes = dict(evaluation.recourse_removes)
        sol.post_recourse = dict(evaluation.post_recourse_state)
        sol.repaired_state = dict(evaluation.repaired_state)
        sol.repair_adds = dict(evaluation.repair_adds)

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        slot_start = time.time()

        context = self._context_features(t=t, A_prev=A_prev)
        selected_name = self._choose_solver_no_peek(t=t, x=context)

        solver = self.candidate_solvers[selected_name]
        one_start = time.time()
        selected = solver.solve_slot(t, copy.deepcopy(A_prev))
        elapsed = time.time() - one_start
        if float(getattr(selected, "solve_time", 0.0)) <= 0.0:
            selected.solve_time = elapsed

        selected_eval = self.robust_evaluator.evaluate_placement(
            t=t,
            state_current=selected.states,
            state_prev=A_prev,
        )
        self._apply_eval_to_solution(selected, selected_eval, A_prev)

        self._update_after_decision(
            selected_name=selected_name,
            context=context,
            selected_obj=float(selected_eval.objective_blended),
            A_prev=A_prev,
            t=t,
        )

        selected.solve_time = time.time() - slot_start
        repair_tag = "|repaired" if bool(getattr(selected_eval, "repair_made", False)) else ""
        selected.status = f"Feasible|selected={selected_name}{repair_tag}"

        if bool(getattr(self.config, "verbose", True)):
            rolling = {
                name: self._rolling_mean(self.history[name])
                for name in self.candidate_names
            }
            rolling_msg = ", ".join(f"{k}:{v:.4f}" for k, v in rolling.items())
            failed = [j for j, v in selected.failures.items() if v > 0]
            print(
                f"[Ensemble t={t}] selected={selected_name} "
                f"slot_obj={float(selected.objective_value):.4f} "
                f"wc_obj={float(selected_eval.objective_worst):.4f} "
                f"fails={failed if failed else '-'} "
                f"method={self.bandit_method} "
                f"rolling({self.window})=[{rolling_msg}]"
            )

        return selected
