# Agent 1 — Research & Planning Agent

Owns Task 2 and Task 5. Deliver depth research, architecture, and dev-ready specs for basic and learnable topological features.

Links: `.taskmaster/tasks/task_002.txt`, `.taskmaster/tasks/task_005.txt`, `.taskmaster/tasks/tasks.json`

## Objectives

- Task 2: Specify deterministic vectorizers and storage schemas that are reproducible and query-efficient.
- Task 5: Specify learnable TDA layers with interfaces compatible with PyTorch and downstream models.

## Success Criteria

- Written specs unlock engineering with minimal back-and-forth.
- Benchmarked reference configurations and datasets selected with acceptance metrics aligned to task files.
- Risks identified with mitigations and fallback options.

---

## Task 2 — Basic Topological Feature Vectorization and Storage

### Scope

- Persistence landscapes, persistence images, Betti curves (FR-CORE-002)
- Storage for: diagrams, barcodes, vectorized features (ST-301)
- Indexing, versioning, similarity search hooks

### Research Checklist

- Methods: Bubenik landscapes; Adams/Atienza images; Betti curves; stability properties and hyperparameters
- Libraries: GUDHI, giotto-tda, ripser/gudhi bindings (licensing, performance)
- Similarity metrics: p-Wasserstein, bottleneck; kernels on diagrams; approximate NN options

### Reference Specs (Dev-Ready)

- Vectorizers API (Python binding calling C++ core outputs):
```python
class Vectorizers:
    def landscapes(diagrams, k_max: int, grid: int, norm: str) -> np.ndarray: ...
    def images(diagrams, resolution: tuple[int,int], sigma: float, weighting: str) -> np.ndarray: ...
    def betti_curves(barcodes, grid: int, dims: list[int]) -> np.ndarray: ...
```

- Storage: PostgreSQL (metadata, indexing) + MongoDB (raw diagrams) dual-store
  - PostgreSQL schema (key tables/indices):
```sql
-- versions
CREATE TABLE tda_feature_version (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  spec_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
-- entities
CREATE TABLE tda_entity (
  id BIGSERIAL PRIMARY KEY,
  domain TEXT NOT NULL,           -- finance|cyber|benchmark
  source TEXT NOT NULL,           -- dataset or asset symbol
  window_start TIMESTAMPTZ,
  window_end TIMESTAMPTZ
);
-- features
CREATE TABLE tda_feature (
  id BIGSERIAL PRIMARY KEY,
  entity_id BIGINT REFERENCES tda_entity(id),
  version_id BIGINT REFERENCES tda_feature_version(id),
  kind TEXT NOT NULL,             -- diagram|barcode|landscape|image|betti
  dims INT[] NOT NULL,
  hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON tda_feature (kind);
CREATE INDEX ON tda_feature USING gin (dims);
CREATE UNIQUE INDEX tda_feature_unique ON tda_feature(hash);
```
  - MongoDB collections:
    - `diagrams`: { feature_id, dim, points: [[b,d], ...] }
    - `barcodes`: { feature_id, dim, intervals: [[b,d], ...] }
    - `vectors`: { feature_id, vector: float32[...], dtype, shape }
  - Versioning: `spec_hash` computed from full parameter JSON of the pipeline

- Similarity search hooks:
```json
{
  "feature_id": 123,
  "vector_l2": true,
  "diagram_metric": "wasserstein",
  "p": 2,
  "ground_cost": "euclidean"
}
```
Expose via future API for k-NN queries. Plan for PGVector/FAISS option.

### Data/Benchmarks

- Synthetic shapes (circle/torus) for correctness; `Topobench` datasets for scale
- Store 100k feature rows as performance smoke test: write TPS, p50/p99 latency

### Acceptance Checklist

- Deterministic outputs: hash-stable given same input/spec
- Schemas created; indices present; CRUD validated
- Query: retrieve by domain/source/time; similarity hook documented
- Storage write p95 < 50ms; read p95 < 30ms on test rig (config to be documented)

### Risks/Mitigations

- Large diagrams: compress with quantization; offload to Mongo with capped collections
- Metric cost: pre-embed with landscape norms for coarse filtering

---

## Task 5 — Advanced Learnable Topological Features

### Scope

- Base `TDALayer` + three concrete layers: Persistence Attention, Learnable Hierarchical Clustering, TDA-GNN
- Seamless PyTorch integration; gradient flow verified

### Research Checklist

- Differentiable PD embeddings; attention on sets; stability/regularization
- GNN libs: PyTorch Geometric; batching and dynamic graphs
- Datasets: ModelNet10/40, synthetic graphs; metrics and baselines

### Reference Specs (Dev-Ready)

- Base layer contract:
```python
class TDALayer(nn.Module):
    def forward(self, diagrams: list[Tensor], metadata: dict|None=None) -> Tensor: ...
    # diagrams: list length = batch, each Tensor [num_points, 3] with (birth, death, dim)
```

- Persistence Attention layer:
  - Input: batch of PD tensors, mask support
  - Mechanism: self-attention over PD points with learned projection of (b,d,dim, persistence)
  - Output: fixed-size embedding `Tensor[batch, d_model]`

- Learnable Hierarchical Clustering layer:
  - Input: linkage-like structure or 0D PD; outputs pooled embedding via learnable merges

- TDA-GNN embedding layer:
  - Construct 1-skeleton at chosen ε; run GCN/GAT; global pooling to vector

- Training/Testing Protocol
  - Unit tests: shapes, gradients, determinism for seeded inputs
  - Integration: plug into small classifier; verify improvement over baseline

### Acceptance Checklist

- Layers instantiate and run forward/backward with dummy PDs
- Gradients non-zero for trainable params; no NaNs
- Reference notebook demonstrating usage with one public dataset
- Configs documented: d_model, heads, ε construction, pooling strategy

### Risks/Mitigations

- Variable-size PDs: masking and padding; set transformers as backup
- Graph build blow-up: sparsify with k-NN or radius pruning
