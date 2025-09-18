# hone.py

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Set

Node = int
Edge = Tuple[Node, Node]
Triangle = Tuple[Node, Node, Node]


def normalize_edges(edges: Iterable[Edge]) -> List[Edge]:
    norm = {(u, v) if u < v else (v, u) for u, v in edges if u != v}
    return sorted(norm)


def build_adjacency(edges: Iterable[Edge]) -> Dict[Node, Set[Node]]:
    """Build undirected adjacency list."""
    adj: Dict[Node, Set[Node]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def normalize_triangles(triangles: Iterable[Triangle]) -> List[Triangle]:
    """Return sorted unique triangles as (a<b<c)."""
    ts = {tuple(sorted(t)) for t in triangles if len(set(t)) == 3}
    return sorted(ts)


def index_triangles_by_node(triangles: Iterable[Triangle]) -> Dict[Node, List[Triangle]]:
    """Map node -> list of triangles containing the node."""
    idx: Dict[Node, List[Triangle]] = defaultdict(list)
    for a, b, c in triangles:
        tri = (a, b, c)
        idx[a].append(tri)
        idx[b].append(tri)
        idx[c].append(tri)
    return idx


def enumerate_higher_order_neighbors(
    edges: Iterable[Edge],
    triangles: Iterable[Triangle],
    *,
    deduplicate: bool = True,
    include_self_if_intersection: bool = True,
) -> Dict[Node, List[Triangle]]:
    """
    For each node u, return higher-order neighbor triangles reachable via incident edges.

    Semantics:
      - For each incident edge (u, v), consider all triangles T that contain v.
      - Exclude triangles T that also contain u, UNLESS u belongs to >= 2 triangles
        (intersection node) and include_self_if_intersection=True.
      - If deduplicate=True, each triangle appears at most once per node.

    Returns:
      Dict[node, List[triangle]] with triangles as sorted tuples.
    """
    E = normalize_edges(edges)
    T = normalize_triangles(triangles)

    if not E:
        return {}

    adj = build_adjacency(E)
    tri_idx = index_triangles_by_node(T)

    # Collect all nodes appearing in edges or triangles
    nodes: Set[Node] = set()
    for u, v in E:
        nodes.add(u)
        nodes.add(v)
    for a, b, c in T:
        nodes.update((a, b, c))

    result: Dict[Node, List[Triangle]] = {u: [] for u in nodes}

    for u in nodes:
        incident = adj.get(u, set())
        u_tri_count = len(tri_idx.get(u, []))
        allow_self = include_self_if_intersection and (u_tri_count >= 2)

        collector: List[Triangle] = []
        seen: Set[Triangle] = set()

        for v in incident:
            for tri in tri_idx.get(v, []):
                if (u in tri) and not allow_self:
                    continue
                if deduplicate:
                    if tri not in seen:
                        seen.add(tri)
                        collector.append(tri)
                else:
                    collector.append(tri)

        result[u] = collector

    return result


if __name__ == "__main__":
    edges = []
    triangles = []

    hone = enumerate_higher_order_neighbors(edges, triangles)
    for u in sorted(hone):
        print(u, hone[u])
