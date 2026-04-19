from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pulp

from objective_calculator import ObjectiveCalculator
from milp_solver import MILPSolver


@dataclass
class RobustEvaluation:
    objective_nominal_raw: float = 0.0
    R_nominal_raw: float = 0.0
    B_nominal_raw: float = 0.0
    Op_cost_raw: float = 0.0
    objective_blended: float = 0.0
    objective_nominal: float = 0.0
    objective_worst: float = 0.0
    objective_worst_raw: float = 0.0
    objective_worst_repaired: float = 0.0
    R_nominal: float = 0.0
    B_nominal: float = 0.0
    Op_cost: float = 0.0
    R_wc: float = 0.0
    B_wc: float = 0.0
    worst_failures_raw: Dict[int, int] = field(default_factory=dict)
    worst_failures: Dict[int, int] = field(default_factory=dict)
    recourse_removes: Dict[Tuple[int, int], int] = field(default_factory=dict)
    post_recourse_state: Dict[Tuple[int, int], int] = field(default_factory=dict)
    repaired_state: Dict[Tuple[int, int], int] = field(default_factory=dict)
    repair_adds: Dict[Tuple[int, int], int] = field(default_factory=dict)
    repair_made: bool = False


class RobustEvaluator:
    """
        Shared robustness evaluator for meta/non-MILP usage.

        Robust-mode methodology is intentionally aligned with MILP by delegating
        worst-case evaluation to MILPSolver's adversarial subproblem for a fixed x.
    """

    def __init__(self, config, data_gen):
        self.config = config
        self.data = data_gen
        self.calculator = ObjectiveCalculator(config, data_gen)
        self._milp_oracle = MILPSolver(config, data_gen)

    def _normalize_failures(self, failures_sparse: Dict[int, int]) -> Dict[int, int]:
        out = {j: 0 for j in range(self.config.num_servers)}
        for j, v in failures_sparse.items():
            out[int(j)] = 1 if int(v) > 0 else 0
        return out

    def _active_datasets(self, t: int) -> List[int]:
        return [
            i
            for i in range(self.config.num_datasets)
            if self.data.active_datasets.get((i, t), 0) == 1
        ]

    def _current_server_loads(self, state_current: Dict[Tuple[int, int], int]) -> Dict[int, float]:
        loads = {j: 0.0 for j in range(self.config.num_servers)}
        for (i, j), v in state_current.items():
            if v > 0:
                loads[j] += float(self.config.dataset_sizes[i])
        return loads

    def _all_servers(self) -> List[int]:
        return list(range(self.config.num_servers))

    def _robust_replica_count(
        self,
        t: int,
        state_current: Dict[Tuple[int, int], int],
        i: int,
        p: int,
    ) -> int:
        h_i = self.config.hop_budgets[i]
        return sum(
            1
            for j in range(self.config.num_servers)
            if state_current.get((i, j), 0) == 1 and self.config.hop_distances.get((j, p), float("inf")) <= h_i
        )

    def _repair_to_robust_feasible(
        self,
        t: int,
        state_current: Dict[Tuple[int, int], int],
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], bool]:
        """
        Greedy repair for K=1-style robustness: ensure each active (i,p) has at least
        K+1 replicas in its hop neighborhood whenever the AP is alive.
        """
        repaired = dict(state_current)
        repair_adds: Dict[Tuple[int, int], int] = {}
        changed = False

        if not bool(getattr(self.config, "use_robust", False)):
            return repaired, repair_adds, changed

        if not bool(getattr(self.config, "use_repair_in_robust", True)):
            return repaired, repair_adds, changed

        k_needed = int(self.config.K_failures) + 1
        if k_needed <= 1:
            return repaired, repair_adds, changed

        active = self._active_datasets(t)
        p_t = self.data.attachment_points.get(t, [])

        # ------------------------------------------------------------------
        # Exact repair MILP: keep all existing replicas, add the minimum-cost
        # extra replicas needed so each active (i,p) has at least K+1 copies
        # in its hop neighborhood.
        # ------------------------------------------------------------------
        repair_model = pulp.LpProblem(f"RobustRepair_t{t}", pulp.LpMinimize)
        x = pulp.LpVariable.dicts(
            "x_repair",
            [(i, j) for i in range(self.config.num_datasets) for j in range(self.config.num_servers)],
            cat="Binary",
        )
        add = pulp.LpVariable.dicts(
            "add_repair",
            [(i, j) for i in range(self.config.num_datasets) for j in range(self.config.num_servers)],
            cat="Binary",
        )

        for i in range(self.config.num_datasets):
            for j in range(self.config.num_servers):
                current_val = int(state_current.get((i, j), 0))
                if current_val == 1:
                    repair_model += x[(i, j)] == 1
                    repair_model += add[(i, j)] == 0
                else:
                    repair_model += x[(i, j)] == add[(i, j)]

        # Keep the final placement within capacity.
        for j in range(self.config.num_servers):
            repair_model += pulp.lpSum(
                self.config.dataset_sizes[i] * x[(i, j)] for i in range(self.config.num_datasets)
            ) <= self.config.server_capacities.get(j, float("inf"))

        # Sufficient robust coverage: K+1 copies in each hop neighborhood.
        for i in active:
            h_i = self.config.hop_budgets[i]
            for p in p_t:
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue
                neighborhood = self.data.get_neighborhood(p, h_i)
                if len(neighborhood) < k_needed:
                    continue
                repair_model += pulp.lpSum(x[(i, j)] for j in neighborhood) >= k_needed

        # Minimize weighted add cost.
        repair_model += pulp.lpSum(
            self.config.get_add_cost(i, j, t) * add[(i, j)]
            for i in range(self.config.num_datasets)
            for j in range(self.config.num_servers)
        )

        repair_solver = pulp.PULP_CBC_CMD(
            timeLimit=int(getattr(self.config, "solver_time_limit", 60)),
            gapRel=float(getattr(self.config, "solver_gap", 0.01)),
            msg=0,
        )
        repair_model.solve(repair_solver)
        repair_status = pulp.LpStatus[repair_model.status]
        if repair_status in ["Optimal", "Feasible"]:
            for i in range(self.config.num_datasets):
                for j in range(self.config.num_servers):
                    if pulp.value(x[(i, j)]) > 0.5:
                        repaired[(i, j)] = 1
                    else:
                        repaired[(i, j)] = 0
                    if pulp.value(add[(i, j)]) > 0.5:
                        repair_adds[(i, j)] = 1
            return repaired, repair_adds, True

        active = self._active_datasets(t)
        p_t = self.data.attachment_points.get(t, [])
        loads = self._current_server_loads(repaired)
        capacities = {j: float(self.config.server_capacities.get(j, float("inf"))) for j in range(self.config.num_servers)}

        for i in active:
            h_i = self.config.hop_budgets[i]
            for p in p_t:
                if self.data.counts.get((i, p, t), 0) <= 0:
                    continue

                current_count = self._robust_replica_count(t, repaired, i, p)
                if current_count >= k_needed:
                    continue

                candidate_servers: List[int] = [
                    j
                    for j in range(self.config.num_servers)
                    if repaired.get((i, j), 0) == 0
                    and self.config.hop_distances.get((j, p), float("inf")) <= h_i
                    and loads[j] + float(self.config.dataset_sizes[i]) <= capacities[j]
                ]
                candidate_servers.sort(key=lambda j: (capacities[j] - loads[j], -loads[j]), reverse=True)

                while current_count < k_needed and candidate_servers:
                    j = candidate_servers.pop(0)
                    repaired[(i, j)] = 1
                    repair_adds[(i, j)] = 1
                    loads[j] += float(self.config.dataset_sizes[i])
                    current_count += 1
                    changed = True

        return repaired, repair_adds, changed

    def evaluate_placement(
        self,
        t: int,
        state_current: Dict[Tuple[int, int], int],
        state_prev: Dict[Tuple[int, int], int],
    ) -> RobustEvaluation:
        # Fast path: when robust mode is OFF, skip all robust machinery
        if not bool(getattr(self.config, "use_robust", False)):
            base_nominal = self.calculator.calculate(t=t, state_current=state_current, state_prev=state_prev)
            out = RobustEvaluation(
                R_nominal=float(base_nominal.R_nominal),
                B_nominal=float(base_nominal.B_nominal),
                Op_cost=float(base_nominal.Op_cost),
                objective_nominal=float(base_nominal.objective_nominal),
                post_recourse_state=dict(state_current),
                repaired_state=dict(state_current),
            )
            out.objective_worst = 0.0
            out.objective_blended = float(out.objective_nominal)
            return out

        # Robust path: full two-stage pipeline
        base_raw = self.calculator.calculate(t=t, state_current=state_current, state_prev=state_prev)
        out = RobustEvaluation(
            R_nominal_raw=float(base_raw.R_nominal),
            B_nominal_raw=float(base_raw.B_nominal),
            Op_cost_raw=float(base_raw.Op_cost),
            objective_nominal_raw=float(base_raw.objective_nominal),
            R_nominal=float(base_raw.R_nominal),
            B_nominal=float(base_raw.B_nominal),
            Op_cost=float(base_raw.Op_cost),
            objective_nominal=float(base_raw.objective_nominal),
            post_recourse_state=dict(state_current),
            repaired_state=dict(state_current),
        )

        # Check if repair is enabled
        use_repair = bool(getattr(self.config, "use_repair_in_robust", True))

        repaired_state, repair_adds, repaired = (
            self._repair_to_robust_feasible(t, state_current) if use_repair else (state_current, {}, False)
        )
        out.repaired_state = repaired_state
        out.repair_adds = repair_adds
        out.repair_made = repaired

        exec_state = repaired_state if repaired else state_current

        # Final objective is computed on the executed state (the repaired state if one was needed).
        base_exec = self.calculator.calculate(t=t, state_current=exec_state, state_prev=state_prev)
        out.R_nominal = float(base_exec.R_nominal)
        out.B_nominal = float(base_exec.B_nominal)
        out.Op_cost = float(base_exec.Op_cost)
        out.objective_nominal = float(base_exec.objective_nominal)

        adv = self._milp_oracle._solve_adversarial_subproblem(
            t=t,
            A_prev=state_prev,
            x_fixed=state_current,
        )
        failures = self._normalize_failures(adv.get("failures", {}))
        out.objective_worst_raw = float(adv.get("objective", 0.0))
        out.R_wc = float(adv.get("R_wc", 0.0))
        out.B_wc = float(adv.get("B_wc", 0.0))
        out.worst_failures_raw = failures

        if repaired and use_repair:
            repaired_adv = self._milp_oracle._solve_adversarial_subproblem(
                t=t,
                A_prev=state_prev,
                x_fixed=repaired_state,
            )
            out.objective_worst_repaired = float(repaired_adv.get("objective", out.objective_worst_raw))
            out.R_wc = float(repaired_adv.get("R_wc", out.R_wc))
            out.B_wc = float(repaired_adv.get("B_wc", out.B_wc))
            out.worst_failures = self._normalize_failures(repaired_adv.get("failures", {}))
        else:
            out.objective_worst_repaired = float(out.objective_worst_raw)
            out.worst_failures = dict(out.worst_failures_raw)

        # Report the executed state as the repaired state when repair was needed.
        out.recourse_removes = {}
        out.post_recourse_state = dict(exec_state)
        out.objective_worst = float(out.objective_worst_repaired)
        out.objective_blended = float(
            (1.0 - self.config.rho) * (out.R_nominal + out.B_nominal)
            + self.config.rho * out.objective_worst_repaired
            - out.Op_cost
        )
        return out