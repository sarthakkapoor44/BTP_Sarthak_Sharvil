from typing import Dict, Tuple, List, Set
import math
import time

import numpy as np

from solver_base import BaseSolver, SlotSolution
from objective_calculator import ObjectiveCalculator


class KMeansDatasetSolver(BaseSolver):
    """
    Per-dataset independent clustering solver (ignores capacity/bandwidth coupling):
      1) For each active dataset i, gather demand nodes at slot t.
      2) Start with k = number of demand nodes, then reduce k (binary search or linear).
      3) For each candidate k, pick k centers using a kmeans++-style farthest-first strategy.
      4) Build clusters by nearest center (hop distance).
      5) For each cluster, choose a feasible centroid server whose max hop to the cluster
         is <= H_i (dataset hop budget), minimizing that max hop.
      6) Place dataset only at the selected centroids (extra replicas are removed).
    """

    def __init__(self, config, data):
        super().__init__(config, data)
        self.calculator = ObjectiveCalculator(config, data)

    def _build_shortest_paths(self, J: int) -> np.ndarray:
        shortest_paths = getattr(self.config, "shortest_paths", None)
        if shortest_paths is not None:
            return shortest_paths

        shortest_paths = np.full((J, J), np.inf, dtype=float)
        for j in range(J):
            shortest_paths[j, j] = 0.0

        hop_d = getattr(self.config, "hop_distances", {})
        for (jj, kk), d in hop_d.items():
            jj_i = int(jj)
            kk_i = int(kk)
            if 0 <= jj_i < J and 0 <= kk_i < J:
                try:
                    shortest_paths[jj_i, kk_i] = float(d)
                except Exception:
                    shortest_paths[jj_i, kk_i] = math.inf
        return shortest_paths

    def _demand_nodes(self, i: int, t: int, J: int) -> List[int]:
        nodes: Set[int] = set()
        for p in self.data.attachment_points.get(t, []):
            p_i = int(p)
            if 0 <= p_i < J and float(self.data.counts.get((i, p, t), 0)) > 0:
                nodes.add(p_i)
        return sorted(nodes)

    def _seed_centers(self,
                      demand_nodes: List[int],
                      prev_replicas: List[int],
                      k: int,
                      shortest_paths: np.ndarray) -> List[int]:
        if not demand_nodes or k <= 0:
            return []

        # First seed: node farthest from previous replicas of dataset i,
        # where farthest is measured by SUM of hop distances.
        if prev_replicas:
            first = max(
                demand_nodes,
                key=lambda p: sum(shortest_paths[p, r] for r in prev_replicas),
            )
        else:
            # If no existing replicas, use spread-out initialization on demand nodes.
            first = max(demand_nodes, key=lambda p: sum(shortest_paths[p, q] for q in demand_nodes if q != p))

        centers: List[int] = [first]
        remaining = [p for p in demand_nodes if p != first]

        while len(centers) < k and remaining:
            # Deterministic kmeans++-style spread using SUM distance to current centers.
            nxt = max(remaining, key=lambda p: sum(shortest_paths[p, c] for c in centers))
            centers.append(nxt)
            remaining.remove(nxt)

        return centers

    def _assign_clusters(self,
                         demand_nodes: List[int],
                         centers: List[int],
                         shortest_paths: np.ndarray) -> List[List[int]]:
        if not centers:
            return []

        clusters: List[List[int]] = [[] for _ in centers]
        for p in demand_nodes:
            idx = min(range(len(centers)), key=lambda c_idx: shortest_paths[p, centers[c_idx]])
            clusters[idx].append(p)
        return clusters

    def _cluster_centroid(self,
                          cluster_nodes: List[int],
                          hop_budget: int,
                          J: int,
                          shortest_paths: np.ndarray,
                          prev_replicas: Set[int]) -> Tuple[bool, int]:
        if not cluster_nodes:
            return False, -1

        best_server = -1
        best_key: Tuple[int, float, float] = (1, math.inf, math.inf)

        for s in range(J):
            radius = max(shortest_paths[p, s] for p in cluster_nodes)
            if radius <= hop_budget:
                prefer_new = 0 if s in prev_replicas else 1
                total_dist = sum(shortest_paths[p, s] for p in cluster_nodes)
                candidate_key = (prefer_new, radius, total_dist)
                if candidate_key < best_key:
                    best_key = candidate_key
                    best_server = s

        return (best_server >= 0), best_server

    def _dedup_preserve_order(self, seq: List[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _dedup_no_demand_replicas(self, prev_replicas: List[int]) -> List[int]:
        """Keep at most one existing replica when a dataset has no demand in slot t."""
        if not prev_replicas:
            return []
        # Keep the first existing replica to avoid unnecessary add operations.
        return [int(prev_replicas[0])]

    def _finalize_clusters_with_centroids(self,
                                          centers: List[int],
                                          demand_nodes: List[int],
                                          hop_budget: int,
                                          J: int,
                                          shortest_paths: np.ndarray,
                                          prev_replicas: Set[int]) -> Tuple[bool, List[List[int]], List[int], List[int]]:
        if not centers:
            return False, [], [], []

        clusters = self._assign_clusters(demand_nodes, centers, shortest_paths)
        clusters = [cluster for cluster in clusters if cluster]
        if not clusters:
            return False, [], [], []

        centroid_per_cluster: List[int] = []
        for cluster in clusters:
            feasible, centroid = self._cluster_centroid(
                cluster_nodes=cluster,
                hop_budget=hop_budget,
                J=J,
                shortest_paths=shortest_paths,
                prev_replicas=prev_replicas,
            )
            if not feasible:
                return False, [], [], []
            centroid_per_cluster.append(int(centroid))

        replicas = self._dedup_preserve_order(centroid_per_cluster)
        if not replicas:
            return False, [], [], []

        for p in demand_nodes:
            covered = any(shortest_paths[p, r] <= hop_budget for r in replicas)
            if not covered:
                return False, [], [], []

        return True, clusters, centroid_per_cluster, replicas

    def _construct_replicas_for_k(self,
                                  i: int,
                                  t: int,
                                  k: int,
                                  demand_nodes: List[int],
                                  prev_replicas: List[int],
                                  hop_budget: int,
                                  J: int,
                                  shortest_paths: np.ndarray) -> Tuple[bool, List[int], List[List[int]], List[int]]:
        centers = self._seed_centers(demand_nodes, prev_replicas, k, shortest_paths)
        if not centers:
            return False, [], [], []

        prev_replica_set = set(prev_replicas)
        max_refine_iters = int(getattr(self.config, "kmeans_refine_iters", 8))

        for _ in range(max_refine_iters):
            ok, _, centroid_per_cluster, _ = self._finalize_clusters_with_centroids(
                centers=centers,
                demand_nodes=demand_nodes,
                hop_budget=hop_budget,
                J=J,
                shortest_paths=shortest_paths,
                prev_replicas=prev_replica_set,
            )
            if not ok:
                return False, [], [], []

            new_centers = self._dedup_preserve_order(centroid_per_cluster)
            if not new_centers:
                return False, [], [], []
            if new_centers == centers:
                break
            centers = new_centers

        ok, clusters, centroid_per_cluster, replicas = self._finalize_clusters_with_centroids(
            centers=centers,
            demand_nodes=demand_nodes,
            hop_budget=hop_budget,
            J=J,
            shortest_paths=shortest_paths,
            prev_replicas=prev_replica_set,
        )
        if not ok:
            return False, [], [], []

        return True, replicas, clusters, centroid_per_cluster

    def _minimize_k_replicas(self,
                             i: int,
                             t: int,
                             demand_nodes: List[int],
                             prev_replicas: List[int],
                             hop_budget: int,
                             J: int,
                             shortest_paths: np.ndarray) -> Tuple[List[int], List[List[int]], List[int], int]:
        if not demand_nodes:
            return [], [], [], 0

        k_hi = len(demand_nodes)
        search_mode = str(getattr(self.config, "kmeans_k_search", "binary")).lower()

        best_replicas: List[int] = []
        best_clusters: List[List[int]] = []
        best_cluster_centroids: List[int] = []
        best_k = 0

        if search_mode == "linear":
            for k in range(k_hi, 0, -1):
                ok, reps, clusters, cluster_centroids = self._construct_replicas_for_k(
                    i, t, k, demand_nodes, prev_replicas, hop_budget, J, shortest_paths
                )
                if ok:
                    best_replicas = reps
                    best_clusters = clusters
                    best_cluster_centroids = cluster_centroids
                    best_k = k
                else:
                    break
            return best_replicas, best_clusters, best_cluster_centroids, best_k

        # Default: binary search for minimum feasible k.
        lo, hi = 1, k_hi
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, reps, clusters, cluster_centroids = self._construct_replicas_for_k(
                i, t, mid, demand_nodes, prev_replicas, hop_budget, J, shortest_paths
            )
            if ok:
                best_replicas = reps
                best_clusters = clusters
                best_cluster_centroids = cluster_centroids
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return best_replicas, best_clusters, best_cluster_centroids, best_k

    def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
        tic = time.time()

        I = self.config.num_datasets
        J = self.config.num_servers

        shortest_paths = self._build_shortest_paths(J)
        verbose = bool(getattr(self.config, "verbose", True))

        # Start from previous placement, then rewrite per active dataset independently.
        current_storage = np.zeros((J, I), dtype=bool)
        for (i, j), v in A_prev.items():
            if v and 0 <= i < I and 0 <= j < J:
                current_storage[j, i] = True

        if verbose:
            print(f"[kmeans_dataset] slot={t} cluster summary")

        for i in range(I):
            if self.data.active_datasets.get((i, t), 0) != 1:
                continue

            prev_replicas = [j for j in range(J) if current_storage[j, i]]

            demand_nodes = self._demand_nodes(i, t, J)
            if not demand_nodes:
                # No demand: deduplicate to one copy, removing replicas that serve no purpose this slot.
                keep_replicas = self._dedup_no_demand_replicas(prev_replicas)
                current_storage[:, i] = False
                for r in keep_replicas:
                    current_storage[r, i] = True
                if verbose:
                    print(f"  dataset={i} no demand, dedup replicas {prev_replicas} -> {keep_replicas}")
                continue

            hop_budget = int(self.config.hop_budgets.get(i, 1))

            best_replicas, best_clusters, best_cluster_centroids, best_k = self._minimize_k_replicas(
                i=i,
                t=t,
                demand_nodes=demand_nodes,
                prev_replicas=prev_replicas,
                hop_budget=hop_budget,
                J=J,
                shortest_paths=shortest_paths,
            )

            if verbose:
                if best_replicas:
                    print(
                        f"  dataset={i} k={best_k} clusters={len(best_clusters)} replicas={best_replicas}"
                    )
                    for c_idx, cluster_nodes in enumerate(best_clusters):
                        centroid = best_cluster_centroids[c_idx] if c_idx < len(best_cluster_centroids) else -1
                        print(f"    cluster#{c_idx}: nodes={cluster_nodes} centroid={centroid}")
                else:
                    print(
                        f"  dataset={i} no feasible clustering found, keeping previous replicas={prev_replicas}"
                    )

            # If no feasible clustering found, keep previous replicas (safe fallback).
            if best_replicas:
                current_storage[:, i] = False
                for r in best_replicas:
                    current_storage[r, i] = True

        state_current = {(i, j): 1 if current_storage[j, i] else 0 for i in range(I) for j in range(J)}
        breakdown = self.calculator.calculate(t=t, state_current=state_current, state_prev=A_prev)

        sol = SlotSolution(time_slot=t, status="Feasible")

        adds: Dict[Tuple[int, int], int] = {}
        removes: Dict[Tuple[int, int], int] = {}
        for i in range(I):
            for j in range(J):
                prev = 1 if A_prev.get((i, j), 0) else 0
                now = 1 if current_storage[j, i] else 0
                if prev == 0 and now == 1:
                    adds[(i, j)] = 1
                elif prev == 1 and now == 0:
                    removes[(i, j)] = 1

        sol.adds = adds
        sol.removes = removes
        sol.states = state_current

        sol.R_nominal = float(breakdown.R_nominal)
        sol.B_nominal = float(breakdown.B_nominal)
        sol.R_wc = float(breakdown.R_wc)
        sol.B_wc = float(breakdown.B_wc)
        sol.Op_cost = float(breakdown.Op_cost)
        sol.objective_value = (breakdown.objective_total if self.config.use_robust else breakdown.objective_nominal)

        sol.solve_time = time.time() - tic
        return sol
