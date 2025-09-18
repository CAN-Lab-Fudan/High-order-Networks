# enumerate_simplices.py
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple

Node = int
Edge = Tuple[Node, Node]
Clique = Tuple[Node, ...]


def normalize_edges(edges: Iterable[Edge]) -> List[Edge]:
    """
    Normalize undirected edges:
    - Remove self-loops (u != v)
    - Ensure ordering (min(u, v), max(u, v))
    - Remove duplicates
    - Return sorted list for reproducibility
    """
    norm = { (u, v) if u < v else (v, u) for u, v in edges if u != v }
    return sorted(norm)


def build_adjacency(edges: Iterable[Edge]) -> Dict[Node, Set[Node]]:
    """
    Build adjacency list (undirected graph).
    Each neighbor set is unique and allows fast intersection.
    """
    adj: Dict[Node, Set[Node]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def grow_cliques_by_one(adj: Dict[Node, Set[Node]], cliques_k: List[Clique]) -> List[Clique]:
    """
    Expand k-cliques to (k+1)-cliques using common neighbors.
    Only add nodes greater than the last element in the clique to avoid duplicates.
    """
    next_cliques: List[Clique] = []
    for c in cliques_k:
        common = set(adj[c[0]])
        for u in c[1:]:
            common &= adj[u]
        for w in sorted(x for x in common if x > c[-1]):
            next_cliques.append((*c, w))
    return next_cliques


def enumerate_all_simplices(edges: Iterable[Edge]) -> Dict[int, List[Clique]]:
    """
    Enumerate all cliques (simplices) in an undirected graph.

    Returns:
        Dict[int, List[Clique]] mapping k -> list of k-node cliques
        Note: simplex dimension = k - 1
              Example: k=2 -> edges (1-simplices)
                       k=3 -> triangles (2-simplices)
                       k=4 -> tetrahedra (3-simplices)
    """
    E = normalize_edges(edges)
    if not E:
        return {}

    adj = build_adjacency(E)

    cliques_by_k: Dict[int, List[Clique]] = {2: [e for e in E]}
    current = cliques_by_k[2]

    k = 2
    while current:
        next_level = grow_cliques_by_one(adj, current)
        if not next_level:
            break
        k += 1
        cliques_by_k[k] = next_level
        current = next_level

    return cliques_by_k


def as_dimension_dict(cliques_by_k: Dict[int, List[Clique]]) -> Dict[int, List[Clique]]:
    """
    Convert {k: cliques} into {dimension d = k-1: cliques}.
    """
    return { (k - 1): v for k, v in cliques_by_k.items() }


if __name__ == "__main__":
    # Example usage
    sample_edges = [
        (0, 1), (0, 2), (1, 2),      # triangle 0-1-2
        (0, 3), (1, 3), (2, 3),      # tetrahedron 0-1-2-3
        (3, 4), (4, 5), (3, 5),      # triangle 3-4-5
        (2, 4)                       # extra link
    ]

    cliques_by_k = enumerate_all_simplices(sample_edges)
    cliques_by_dim = as_dimension_dict(cliques_by_k)

    print("By k (number of nodes):")
    for k, cliques in cliques_by_k.items():
        print(f"k={k}, count={len(cliques)} -> {cliques}")


