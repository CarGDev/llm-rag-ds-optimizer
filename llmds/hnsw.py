"""HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search.

Implementation based on:
    Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest 
    neighbor search using Hierarchical Navigable Small World graphs. IEEE transactions 
    on pattern analysis and machine intelligence, 42(4), 824-836.

See docs/CITATIONS.md for full citation details.
"""

import random
from typing import Any, Optional

import numpy as np


class HNSW:
    """
    Hierarchical Navigable Small World graph for approximate nearest neighbor search.

    Implements HNSW with configurable M, efConstruction, and efSearch parameters.
    
    Reference:
        Malkov & Yashunin (2018). Efficient and robust approximate nearest neighbor 
        search using Hierarchical Navigable Small World graphs.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float = 1.0 / np.log(2.0),
        seed: Optional[int] = None,
    ):
        """
        Initialize HNSW index.

        Args:
            dim: Dimension of vectors
            M: Maximum number of connections for each node
            ef_construction: Size of candidate set during construction
            ef_search: Size of candidate set during search
            ml: Normalization factor for level assignment
            seed: Optional random seed for reproducible level assignments.
                  If None, uses the global random state.
        """
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml

        # Instance-level random state for reproducibility
        self._rng = random.Random(seed) if seed is not None else random

        # Layers: list of graphs, each graph is dict[node_id] -> list[neighbor_ids]
        self._layers: list[dict[int, list[int]]] = []
        self._vectors: dict[int, np.ndarray] = {}  # node_id -> vector
        self._max_level: dict[int, int] = {}  # node_id -> max level
        self._entry_point: Optional[int] = None
        self._entry_level = 0

    def _random_level(self) -> int:
        """Generate random level for new node."""
        level = 0
        while self._rng.random() < np.exp(-self.ml) and level < 10:
            level += 1
        return level

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2 distance between two vectors."""
        return float(np.linalg.norm(a - b))

    def _search_layer(
        self,
        query: np.ndarray,
        k: int,
        entry_points: list[int],
        layer: dict[int, list[int]],
    ) -> list[tuple[int, float]]:
        """
        Search in a single layer using greedy search.

        Args:
            query: Query vector
            k: Number of results to return
            entry_points: Starting points for search
            layer: Graph layer to search

        Returns:
            List of (node_id, distance) tuples
        """
        if not entry_points:
            return []

        candidates: list[tuple[float, int]] = []
        visited = set(entry_points)
        best_candidates: list[tuple[float, int]] = []

        # Initialize candidates with entry points
        for ep in entry_points:
            if ep in self._vectors:
                dist = self._distance(query, self._vectors[ep])
                candidates.append((dist, ep))
                best_candidates.append((dist, ep))

        # Sort by distance
        candidates.sort()
        best_candidates.sort()

        # Greedy search
        while candidates:
            dist, current = candidates.pop(0)

            # Explore neighbors
            if current in layer:
                for neighbor in layer[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if neighbor in self._vectors:
                            neighbor_dist = self._distance(query, self._vectors[neighbor])
                            candidates.append((neighbor_dist, neighbor))
                            best_candidates.append((neighbor_dist, neighbor))

            # Maintain top-ef_search candidates
            candidates.sort()
            if len(candidates) > self.ef_search:
                candidates = candidates[: self.ef_search]

        # Sort best candidates and return top-k as (node_id, distance) tuples
        best_candidates.sort()
        results = [(node_id, dist) for dist, node_id in best_candidates[:k]]
        return results

    def add(self, vec: np.ndarray, vec_id: int) -> None:
        """
        Add a vector to the index.

        Args:
            vec: Vector to add (must be of dimension self.dim)
            vec_id: Unique identifier for the vector
        """
        if vec.shape != (self.dim,):
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vec.shape[0]}")

        if vec_id in self._vectors:
            raise ValueError(f"Vector ID {vec_id} already exists")

        self._vectors[vec_id] = vec.copy()
        level = self._random_level()
        self._max_level[vec_id] = level

        # Ensure we have enough layers
        while len(self._layers) <= level:
            self._layers.append({})

        # If this is the first node, set as entry point
        if self._entry_point is None:
            self._entry_point = vec_id
            self._entry_level = level
            for l in range(level + 1):
                self._layers[l][vec_id] = []
            return

        # Search for nearest neighbors at each level
        entry_points = [self._entry_point]

        # Start from top layer and work down
        for l in range(min(level, self._entry_level), -1, -1):
            # Search layer for candidates
            candidates = self._search_layer(
                vec, self.ef_construction, entry_points, self._layers[l]
            )
            entry_points = [node_id for node_id, _ in candidates]

        # Insert at all levels up to node's level
        for l in range(min(level, len(self._layers) - 1) + 1):
            if l == 0:
                # Bottom layer: connect to M neighbors
                candidates = self._search_layer(vec, self.M, entry_points, self._layers[l])
            else:
                # Upper layers: connect to M neighbors
                candidates = self._search_layer(vec, self.M, entry_points, self._layers[l])

            # Create connections
            neighbors = [node_id for node_id, _ in candidates[: self.M]]

            if vec_id not in self._layers[l]:
                self._layers[l][vec_id] = []

            # Add bidirectional connections
            for neighbor in neighbors:
                if neighbor not in self._layers[l]:
                    self._layers[l][neighbor] = []
                self._layers[l][vec_id].append(neighbor)
                self._layers[l][neighbor].append(vec_id)

                # Limit connections to M
                if len(self._layers[l][neighbor]) > self.M:
                    # Remove farthest connection
                    neighbor_vec = self._vectors[neighbor]
                    distances = [
                        (self._distance(self._vectors[n], neighbor_vec), n)
                        for n in self._layers[l][neighbor]
                    ]
                    distances.sort(reverse=True)
                    farthest = distances[0][1]
                    self._layers[l][neighbor].remove(farthest)
                    if farthest in self._layers[l]:
                        self._layers[l][farthest].remove(neighbor)

            # Limit connections for new node
            if len(self._layers[l][vec_id]) > self.M:
                distances = [
                    (self._distance(self._vectors[n], vec), n) for n in self._layers[l][vec_id]
                ]
                distances.sort()
                self._layers[l][vec_id] = [n for _, n in distances[: self.M]]

            entry_points = neighbors

        # Update entry point if necessary
        if level > self._entry_level:
            self._entry_point = vec_id
            self._entry_level = level

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of results to return

        Returns:
            List of (vector_id, distance) tuples sorted by distance
        """
        if self._entry_point is None:
            return []

        if query.shape != (self.dim,):
            raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {query.shape[0]}")

        # Start from top layer
        current = self._entry_point
        current_level = self._entry_level

        # Navigate down to level 0
        for l in range(current_level, 0, -1):
            if current not in self._layers[l]:
                continue

            # Find nearest neighbor in this layer
            neighbors = self._layers[l].get(current, [])
            if not neighbors:
                continue

            best_dist = self._distance(query, self._vectors[current])
            best_node = current

            for neighbor in neighbors:
                if neighbor in self._vectors:
                    dist = self._distance(query, self._vectors[neighbor])
                    if dist < best_dist:
                        best_dist = dist
                        best_node = neighbor

            current = best_node

        # Search layer 0
        results = self._search_layer(query, k, [current], self._layers[0])
        return results

    def stats(self) -> dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        total_edges = sum(sum(len(neighbors) for neighbors in layer.values()) for layer in self._layers)
        return {
            "num_vectors": len(self._vectors),
            "num_layers": len(self._layers),
            "entry_point": self._entry_point,
            "entry_level": self._entry_level,
            "total_edges": total_edges,
            "avg_degree": total_edges / len(self._vectors) if self._vectors else 0.0,
        }
