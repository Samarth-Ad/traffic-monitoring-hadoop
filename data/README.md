# data/

Project data lives here and is mounted into every container at `/data`.

Layout (populated at runtime):

```
data/
├── raw/              # Kaggle Bangalore traffic CSV
├── stream/           # streamed batches written by the Phase-2 producer
├── analytics/        # Spark output (Parquet) from Phase 3
└── graph/            # OSMnx GraphML + Phase-5 algorithm results
```

To populate `raw/`:
```
pip install kaggle
# place ~/.kaggle/kaggle.json
python python/download_dataset.py
```
