# Agent 2 — Infrastructure Planning Agent

Owns Task 6 and Task 7. Deliver domain research, pipeline specs, and evaluation plans for finance regime detection and cybersecurity anomaly detection.

Links: `.taskmaster/tasks/task_006.txt`, `.taskmaster/tasks/task_007.txt`, `.taskmaster/tasks/tasks.json`

## Objectives

- Task 6 (Finance): Plan ingestion, preprocessing, feature integration, detection, stability metric, EWS, and backtesting.
- Task 7 (Cyber): Plan streaming ingest, windowing, topological features, baseline profiling, anomaly detection, supervised attack classification, and validation.

## Shared Operating Specs

- Sliding window embedding parameters must be explicitly versioned in feature requests to the core/vectorizers.
- All interfaces defined as contracts with example payloads.

---

## Task 6 — Finance Module for Market Regime Detection

### Data Sources

- Equities: `source=polygon` (or local CSV for offline), `symbol`, `interval`
- Crypto: `source=binance`, `pair`, `interval`

### Preprocessing (Dev-Ready)

- Cleaning: forward-fill small gaps; drop outliers via MAD; normalization per asset
- Sliding window embedding to point clouds:
```json
{
  "window_size": 256,
  "slide": 16,
  "embedding": "delay",
  "delay": 2,
  "features": ["log_return", "volume_z"]
}
```
- Output: `np.ndarray[window, points, dims]`

### Feature Integration Service Contract

- Request to TDA services (Task 1/2/5):
```json
{
  "entity": {"domain": "finance", "source": "polygon", "symbol": "SPY"},
  "window": {"start": "2020-01-01T00:00:00Z", "end": "2020-06-01T00:00:00Z"},
  "vectorizers": {"landscapes": {"k_max": 5, "grid": 256}},
  "learnable": {"tda_gnn": {"epsilon": 0.15, "gnn": "gcn", "hidden": 64}},
  "version": "spec_hash"
}
```
- Response: feature IDs and pointers to storage rows

### Models

- Transition detection (unsupervised):
  - Change-point on landscape norms; PD clustering with Wasserstein linkage
- Regime stability metric:
  - Variance of landscape norm; average PD distance over window
- Early Warning Signals (EWS):
  - Threshold rules on stability slope; z-score of transition likelihood; cool-down period

### Backtesting Plan

- Rolling historical evaluation; align with known events; latency target <100ms per window
- Metrics: transition detection precision/recall, lead time distribution

### Acceptance Checklist

- End-to-end dry-run with CSV data → point clouds → features → transitions → EWS report
- Configurable thresholds and windowing; JSON configs versioned
- Backtest script produces a summary report with KPIs and plots

---

## Task 7 — Cybersecurity Module for Anomaly Detection

### Streaming Ingestion

- Integrate with Kafka topic: `net.packets` (Task 3 dependency)
- Kafka consumer params: group id, at-least-once, JSON payload

### Windowing & Preprocessing

- Time windows with slide; per-window features:
  - Packet sizes, inter-arrival times, 5-tuple features; build point clouds or graph adjacency matrices
- Example config:
```json
{
  "window_ms": 5000,
  "slide_ms": 1000,
  "represent": "graph",
  "graph": {"nodes": "ip:port", "edge_weight": "bytes"}
}
```

### Topological Feature Extraction

- Use Task 1/2/5 services to compute PDs, Betti numbers, learnable embeddings

### Baseline and Detection

- Baseline: train clustering/density model on clean traffic topological signatures
- Deviation detection: Wasserstein distance or model score thresholding with alert hysteresis

### Attack Classification (Supervised)

- Dataset: CIC-IDS2017; labels mapped to windows
- Models: Gradient Boosting or GNN over graph representations augmented with TDA features

### Offline Evaluation

- Full-dataset runs; metrics: FPR < 50%, accuracy > 75%, latency per window

### Acceptance Checklist

- Streaming mock test: replay pcap → Kafka → window → features → detection
- Configurable windowing and thresholds; reproducible via spec hash
- Evaluation notebook/report with metrics table and confusion matrix