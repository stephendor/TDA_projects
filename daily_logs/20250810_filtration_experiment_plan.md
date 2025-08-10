# Filtration Variant Experiment Plan (2025-08-10)

Baseline run: `validation/vector_stack_outputs/20250810_031323`
Manifest hash (baseline): `97de5c7493c154bbe07c990821ad7996dcace6fd1f78c4c48101a76512422649`
SW angle strategy: `golden` (48 angles, hash 8667986c9511b4c4)

## Objectives
1. Evaluate alternative Rips filtration recomputation modes (`dtm`, `sparse`) vs `standard`.
2. Determine if any mode yields meaningful PR AUC lift (>= +0.005) at acceptable cost (<=1.25× train time) or efficiency gains (sparse) with negligible performance loss (<0.003 PR AUC drop).
3. Populate enhanced diagnostics: `filtration_recompute` with full stats and timing subsections.

## Success / Failure Criteria
| Mode | Param | PASS (Candidate) | MARGINAL | FAIL | Special (Sparse) |
|------|-------|------------------|----------|------|------------------|
| dtm  | k     | ΔPR_AUC ≥ +0.005 & time ≤1.25× | +0.003–0.005 (needs confirm) | ΔPR_AUC < +0.003 OR time >1.5× OR feature_dim +>20% w/o gain | n/a |
| sparse | divisor | Same as dtm OR (PR drop <0.003 AND (time ≤0.75× OR feature_dim ≤0.8×)) | n/a | same as dtm | Efficiency win path |

Abort early if first dtm (k=10) AND first sparse (div=10) both FAIL.

## Run Matrix (Phase 1)
Baseline already done.
- dtm k=10 (mid) — first probe
- sparse divisor=10 — first probe
- If any passes early screen: dtm k=5, dtm k=20, sparse 5, sparse 20

## Metrics Captured
- Primary: PR AUC (logistic_regression)
- Secondary: ROC AUC, best F1, Brier, ECE
- Efficiency: logistic train_time_sec ratio vs baseline, feature_dim ratio
- Topology: persistence block presence counts (implicitly via feature blocks), recompute success stats

## Diagnostics Fields Required (filtration_recompute)
```
{
  "attempted_count": int,
  "success_count": int,
  "skipped_no_pointcloud": int,
  "skipped_no_gudhi": int,
  "failed_exception": int,
  "mode": "standard|dtm|sparse",
  "params": {"dtm_k": int? , "sparse_param": int?},
  "mean_points_in": float,
  "mean_points_after_subsample": float,
  "rips_recompute_time_sec": float
}
```

## Procedure
1. Patch extraction script to ensure full diagnostics (if partial).
2. Run dtm k=10 and sparse 10 runs (same flags as baseline except filtration mode/param).
3. Generate deltas using `scripts/enhancement_delta_eval.py --baseline <baseline_dir> --enhanced <run_dir>`.
4. Update this plan file (append results) and prepare summary for decision log.

## Early Abort Logic
After first two variant runs:
- If both ΔPR_AUC < +0.003 and time ratio >1.10 → abort remaining matrix and document failure.

## Seeds & Determinism
Use default pipeline seed (already embedded). Reuse identical manifest; verify hash difference only due to filtration mode inclusion.

## Risks & Mitigations
- Missing point clouds: will cause high `skipped_no_pointcloud` — abort variant (document).
- gudhi import failure: treat as skip with `skipped_no_gudhi` >0 — document.
- Runtime blow-up: monitor after k=10; if >1.5× baseline, do not proceed with k=20.

*Generated automatically.*
