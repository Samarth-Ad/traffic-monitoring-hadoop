"""
Phase 4 — Build the Bangalore road-network graph from OpenStreetMap.

Junctions become nodes, road segments become directed edges. Outputs:

    data/graph/bangalore.graphml   full graph for later algorithms
    data/graph/nodes.parquet       node_id, lat, lon, street_count
    data/graph/edges.parquet       u, v, length_m, highway, name, maxspeed

Run on the host (not in a container) because osmnx needs HTTPS access to the
Overpass API and a reasonable Python toolchain.

    pip install -r graph/requirements.txt
    python graph/build_graph.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd  # noqa: F401  (imported for osmnx side-effects)
import networkx as nx
import osmnx as ox
import pandas as pd

LOG = logging.getLogger("build_graph")

PLACE = "Bangalore, Karnataka, India"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"


def download_graph() -> nx.MultiDiGraph:
    LOG.info("Downloading drivable road network for %s via Overpass...", PLACE)
    graph = ox.graph_from_place(PLACE, network_type="drive", simplify=True)
    LOG.info("Graph has %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def export(graph: nx.MultiDiGraph) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    graphml_path = GRAPH_DIR / "bangalore.graphml"
    ox.save_graphml(graph, filepath=graphml_path)
    LOG.info("Wrote %s", graphml_path)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

    nodes_df = pd.DataFrame({
        "node_id": nodes_gdf.index.astype("int64"),
        "lat": nodes_gdf["y"].astype(float),
        "lon": nodes_gdf["x"].astype(float),
        "street_count": nodes_gdf.get("street_count", pd.Series(dtype="float")).fillna(0).astype(int),
    })
    nodes_df.to_parquet(GRAPH_DIR / "nodes.parquet", index=False)
    LOG.info("Wrote %s (%d rows)", GRAPH_DIR / "nodes.parquet", len(nodes_df))

    edges_df = edges_gdf.reset_index()[["u", "v", "length", "highway", "name", "maxspeed"]].copy()
    edges_df.rename(columns={"length": "length_m"}, inplace=True)
    for col in ("highway", "name", "maxspeed"):
        edges_df[col] = edges_df[col].astype(str)
    edges_df.to_parquet(GRAPH_DIR / "edges.parquet", index=False)
    LOG.info("Wrote %s (%d rows)", GRAPH_DIR / "edges.parquet", len(edges_df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    graph = download_graph()
    export(graph)


if __name__ == "__main__":
    main()
