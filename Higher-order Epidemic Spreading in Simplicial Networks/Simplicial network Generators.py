# simplicial_edge_generators.py

import random
from typing import Dict, Iterable, List, Set, Tuple, Optional

Node = int
Edge = Tuple[Node, Node]
Triangle = Tuple[Node, Node, Node]


# ------------------------------
# Core utilities
# ------------------------------

def _norm_edge(u: Node, v: Node) -> Edge:
    return (u, v) if u < v else (v, u)

def _add_edge(edges: Set[Edge], u: Node, v: Node):
    if u != v:
        edges.add(_norm_edge(u, v))

def make_base_triangles(N2: int, start: int = 0) -> List[Triangle]:
    """
    Build N2 disjoint base 2-simplices (triangles) over contiguous node IDs.
    IMPORTANT: Only these triangles are considered higher-order (2-simplices).
    Any triangles formed by later edges DO NOT count as higher-order structure.
    """
    tris: List[Triangle] = []
    for i in range(N2):
        a, b, c = start + 3 * i, start + 3 * i + 1, start + 3 * i + 2
        tris.append(tuple(sorted((a, b, c))))
    return tris

def triangle_edges(tri: Triangle) -> List[Edge]:
    a, b, c = tri
    return [_norm_edge(a, b), _norm_edge(a, c), _norm_edge(b, c)]


# ------------------------------
# Targets -> probabilities
# ------------------------------

def probs_from_targets(N2: int, ktri_target: float, k_target: float) -> Tuple[float, float]:
    """
    Given targets:
        <k∇> = (N2 - 1) * p∇
        <k>  = <k∇>/3 + 2 + (3*N2 - <k∇>/3 - 3) * p1
    return (p∇, p1) clipped into [0,1].
    """
    if N2 <= 1:
        raise ValueError("N2 must be >= 2")
    p_tri = ktri_target / (N2 - 1)
    denom = (3 * N2 - ktri_target / 3.0 - 3.0)
    p1 = 0.0 if denom <= 0 else (k_target - 2.0 - ktri_target / 3.0) / denom
    p_tri = max(0.0, min(1.0, p_tri))
    p1 = max(0.0, min(1.0, p1))
    return p_tri, p1


# ------------------------------
# (1) Random simplicial network (edges only)
# ------------------------------

def generate_random_simplicial_edges(
    N2: int,
    *,
    p_tri: Optional[float] = None,
    p1: Optional[float] = None,
    ktri_target: Optional[float] = None,
    k_target: Optional[float] = None,
    seed: Optional[int] = None,
) -> Set[Edge]:
    """
    Build edges of a random simplicial network as follows:

      1) Create N2 disjoint base triangles (ONLY these count as 2-simplices).
      2) For each unordered pair of distinct base triangles (i, j):
         with probability p∇, add exactly ONE "seed" edge between a random node of i and a random node of j.
         (These seed edges define the higher-order meta-connections used for <k∇>.)
      3) Over ALL remaining non-edges in the whole graph, add an undirected edge with probability p1.
         NOTE: triangles that may form in this step DO NOT count as higher-order structure.

    Input options:
      - Provide (p_tri, p1) directly, OR provide (ktri_target, k_target) and they will be inverted.

    Output:
      - A set of undirected edges (u, v) with u < v.

    WARNING:
      - Only the base N2 triangles are higher-order 2-simplices. Later edges do NOT create new higher-order simplices.
    """
    rng = random.Random(seed)

    if (ktri_target is not None) and (k_target is not None):
        p_tri, p1 = probs_from_targets(N2, ktri_target, k_target)
    if (p_tri is None) or (p1 is None):
        raise ValueError("Provide either (p_tri, p1) or (ktri_target, k_target).")

    # Base triangles and their intra-edges
    triangles = make_base_triangles(N2, start=0)
    edges: Set[Edge] = set()
    for tri in triangles:
        for e in triangle_edges(tri):
            edges.add(e)

    # Step 2: seed meta-connections
    # (We do NOT return meta data here; this function returns edges only.)
    for i in range(N2):
        ai, bi, ci = triangles[i]
        for j in range(i + 1, N2):
            if rng.random() < p_tri:
                x, y, z = triangles[j]
                u = rng.choice([ai, bi, ci])
                v = rng.choice([x, y, z])
                _add_edge(edges, u, v)

    # Step 3: global 1-simplices with p1 over all remaining non-edges
    n_nodes = 3 * N2
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            e = _norm_edge(u, v)
            if e in edges:
                continue
            if rng.random() < p1:
                edges.add(e)

    return edges


# Optional: diagnostic version that also returns the seed meta-pairs for measuring <k∇>
def generate_random_simplicial_edges_with_meta(
    N2: int,
    *,
    p_tri: Optional[float] = None,
    p1: Optional[float] = None,
    ktri_target: Optional[float] = None,
    k_target: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[Set[Edge], Set[Tuple[int, int]]]:
    """
    Same as generate_random_simplicial_edges, but also returns the set of triangle-pairs (i, j)
    that were connected in Step 2 by a seed edge. Use ONLY these pairs to compute <k∇>.
    """
    rng = random.Random(seed)

    if (ktri_target is not None) and (k_target is not None):
        p_tri, p1 = probs_from_targets(N2, ktri_target, k_target)
    if (p_tri is None) or (p1 is None):
        raise ValueError("Provide either (p_tri, p1) or (ktri_target, k_target).")

    triangles = make_base_triangles(N2, start=0)
    edges: Set[Edge] = set()
    for tri in triangles:
        for e in triangle_edges(tri):
            edges.add(e)

    connected_pairs: Set[Tuple[int, int]] = set()
    for i in range(N2):
        ai, bi, ci = triangles[i]
        for j in range(i + 1, N2):
            if rng.random() < p_tri:
                x, y, z = triangles[j]
                u = rng.choice([ai, bi, ci])
                v = rng.choice([x, y, z])
                _add_edge(edges, u, v)
                connected_pairs.add((i, j))

    n_nodes = 3 * N2
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            e = _norm_edge(u, v)
            if e in edges:
                continue
            if rng.random() < p1:
                edges.add(e)

    return edges, connected_pairs


# ------------------------------
# (2) Star-simplicial network (edges only)
# ------------------------------

def generate_star_simplicial_edges(
    N2: int,
    *,
    seed: Optional[int] = None,
) -> Set[Edge]:
    """
    Build edges of a star-simplicial network of size N2:

      1) Create N2 disjoint base triangles (ONLY these count as 2-simplices).
      2) Choose one triangle uniformly at random as center S.
      3) For each remaining triangle, add exactly ONE edge between a random node in S
         and a random node in that triangle.

    Output:
      - A set of undirected edges (u, v) with u < v.
    """
    rng = random.Random(seed)
    triangles = make_base_triangles(N2, start=0)

    edges: Set[Edge] = set()
    for tri in triangles:
        for e in triangle_edges(tri):
            edges.add(e)

    center = rng.randrange(N2)
    s0, s1, s2 = triangles[center]

    for t in range(N2):
        if t == center:
            continue
        a, b, c = triangles[t]
        u = rng.choice([s0, s1, s2])
        v = rng.choice([a, b, c])
        _add_edge(edges, u, v)

    return edges


