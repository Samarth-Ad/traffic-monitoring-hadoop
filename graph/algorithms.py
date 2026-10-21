"""
Phase 5 — Graph algorithms on the Bangalore road network.

Reads data/graph/bangalore.graphml (produced by build_graph.py) and runs:

  * Dijkstra shortest path (edge weight = length_m)
  * PageRank
  * Betweenness centrality (approximate; k random pivots for speed)
  * Closeness centrality (sample-based)

Writes:

  data/graph/pagerank.parquet         node_id, pagerank
  data/graph/betweenness.parquet      node_id, betweenness
  data/graph/closeness.parquet        node_id, closeness
  data/graph/sample_shortest_path.parquet   origin, dest, length_m, path
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

LOG = logging.getLogger("algorithms")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
GRAPHML = GRAPH_DIR / "bangalore.graphml"

CENTRALITY_SAMPLE_K = int(os.environ.get("CENTRALITY_SAMPLE_K", "200"))
SHORTEST_PATH_PAIRS = int(os.environ.get("SHORTEST_PATH_PAIRS", "25"))


def load_graph() -> nx.MultiDiGraph:
    if not GRAPHML.exists():
        raise FileNotFoundError(f"Missing {GRAPHML}. Run graph/build_graph.py first.")
    LOG.info("Loading %s", GRAPHML)
    return ox.load_graphml(GRAPHML)


def run_pagerank(graph: nx.MultiDiGraph) -> pd.DataFrame:
    LOG.info("Running PageRank...")
    pr = nx.pagerank(graph, weight="length")
    return pd.DataFrame({"node_id": list(pr.keys()), "pagerank": list(pr.values())}).sort_values(
        "pagerank", ascending=False
    )


def run_centrality(graph: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOG.info("Running approximate betweenness (k=%d pivots)...", CENTRALITY_SAMPLE_K)
    k = min(CENTRALITY_SAMPLE_K, graph.number_of_nodes())
    bc = nx.betweenness_centrality(graph, k=k, weight="length", seed=42)
    betweenness = pd.DataFrame(
        {"node_id": list(bc.keys()), "betweenness": list(bc.values())}
    ).sort_values("betweenness", ascending=False)

    LOG.info("Running closeness on %d sampled nodes...", k)
    sampled = random.Random(42).sample(list(graph.nodes), k=k)
    cc = {n: nx.closeness_centrality(graph, u=n, distance="length") for n in sampled}
    closeness = pd.DataFrame(
        {"node_id": list(cc.keys()), "closeness": list(cc.values())}
    ).sort_values("closeness", ascending=False)

    return betweenness, closeness


def sample_shortest_paths(graph: nx.MultiDiGraph) -> pd.DataFrame:
    LOG.info("Computing %d sample shortest paths...", SHORTEST_PATH_PAIRS)
    rng = random.Random(123)
    nodes = list(graph.nodes)
    rows: list[dict] = []
    for _ in range(SHORTEST_PATH_PAIRS):
        origin, dest = rng.sample(nodes, 2)
        try:
            path = nx.shortest_path(graph, origin, dest, weight="length")
            length = nx.shortest_path_length(graph, origin, dest, weight="length")
            rows.append({
                "origin": origin,
                "dest": dest,
                "length_m": float(length),
                "hops": len(path) - 1,
                "path": ",".join(str(p) for p in path[:50]),
            })
        except nx.NetworkXNoPath:
            continue
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    graph = load_graph()

    pagerank = run_pagerank(graph)
    pagerank.to_parquet(GRAPH_DIR / "pagerank.parquet", index=False)
    LOG.info("Wrote pagerank.parquet (%d rows)", len(pagerank))

    betweenness, closeness = run_centrality(graph)
    betweenness.to_parquet(GRAPH_DIR / "betweenness.parquet", index=False)
    closeness.to_parquet(GRAPH_DIR / "closeness.parquet", index=False)
    LOG.info("Wrote betweenness.parquet, closeness.parquet")

    paths = sample_shortest_paths(graph)
    paths.to_parquet(GRAPH_DIR / "sample_shortest_path.parquet", index=False)
    LOG.info("Wrote sample_shortest_path.parquet (%d rows)", len(paths))


if __name__ == "__main__":
    main()
