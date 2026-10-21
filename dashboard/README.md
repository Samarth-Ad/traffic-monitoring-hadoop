# dashboard/ — Phase 6

Streamlit app that visualises the entire pipeline's output. Zero HDFS calls —
everything is read from Parquet in `data/analytics/` and `data/graph/`.

Local run:
```
pip install -r ../docker/requirements.dashboard.txt
streamlit run app.py
```

In Docker (preferred):
```
docker compose up -d dashboard
# http://localhost:8501
```

Views: Overview, Volume, Speed, Congestion, Peak windows, Road network
(Folium map of top-PageRank junctions), Shortest paths.
