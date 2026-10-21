# graph/ — Phases 4 & 5

Phase 4 downloads Bangalore's drivable road network from OpenStreetMap via
OSMnx and persists it as GraphML + Parquet. Phase 5 runs classical graph
algorithms on it. Both run on the host (they need Overpass access, and
OSMnx + geopandas pull a lot of native libs).

## Setup

```
cd graph
pip install -r requirements.txt
python build_graph.py      # Phase 4 — one-shot, writes data/graph/
python algorithms.py       # Phase 5 — writes pagerank/centrality/paths
```

Outputs land in `data/graph/` and are read directly by the Streamlit dashboard.

## Env knobs

* `CENTRALITY_SAMPLE_K`   pivots for approximate betweenness / closeness (default 200)
* `SHORTEST_PATH_PAIRS`   number of random O-D pairs sampled (default 25)
