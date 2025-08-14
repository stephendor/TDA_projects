# Soft-cap Local Merge (Experimental)

Overview:

- The streaming distance matrix supports a soft kNN cap per vertex to reduce work and memory in threshold mode.
- An optional local-merge mode reduces contention in parallel thresholding by using per-block snapshots and a single merge per block.

Status:

- Default: OFF (production). Enable via `--softcap-local-merge 1` in the perf harness or `StreamingDMConfig.softcap_local_merge=true`.
- Experimental: Expect small, bounded overshoot relative to the configured soft cap K in parallel mode.

Telemetry:

- `StreamingDMStats.softcap_overshoot_sum`: cumulative overshoot across vertices during merges.
- `StreamingDMStats.softcap_overshoot_max`: maximum per-vertex overshoot seen.
- The perf harness prints these when `--soft-knn-cap > 0`.

Safeguards and Guidance:

- For race-free, deterministic runs use `--parallel-threshold 0` when `--soft-knn-cap > 0`.
- Use local-merge for higher throughput with `--parallel-threshold 1`; monitor overshoot metrics and adjacency histograms.
- Prefer smaller tiles (`--block`) or periodic resnapshots to further bound overshoot if needed.

QA Tips:

- Run `--baseline-compare 1` to compute a serial baseline (no soft cap) and report deltas in edges and simplices.
- Export adjacency histograms using `--adj-hist-csv` and `--adj-hist-csv-baseline` and compare distributions.
