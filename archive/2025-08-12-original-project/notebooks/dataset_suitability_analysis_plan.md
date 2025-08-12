# Dataset Suitability Analysis Plan for TDA Deep Learning Project

**Created**: August 7, 2025  
**Purpose**: Comprehensive analysis of datasets in `/data/apt_datasets` for TDA-based cybersecurity attack detection  
**Compliance**: Following UNIFIED_AGENT_INSTRUCTIONS.md requirements

## 📋 ANALYSIS REQUIREMENTS FROM INSTRUCTIONS

### Mandatory Pre-Coding Protocol

- **NO EXCEPTIONS**: Coding without prior data analysis is forbidden
- **DATA INTEGRITY**: Explicitly check for temporal overlap and label balance
- **TRACEABILITY**: Link every experiment to its dataset analysis
- **BASELINE ENFORCEMENT**: Compare all results to baseline
- **FAILURE LOGGING**: Document all errors, root causes, and lessons learned

### Dataset Analysis Requirements
1. Load and summarize dataset (feature types, missing values, temporal structure, class balance)
2. Document anomalies, limitations, or risks (e.g., temporal gaps, label imbalance)
3. Save analysis in dedicated notebook/markdown file
4. **Explicitly verify temporal overlap and co-occurrence of attack/benign samples**
5. **Flag any risk of data leakage before proceeding**

## 🎯 PRIMARY DATASETS FOR ANALYSIS

Based on project instructions and current validation results, prioritize:

### Tier 1: Primary Validation Datasets
1. **UNSW-NB15** - Currently validated, production-ready
2. **CIC-IDS-2017** - High TDA relevance, real temporal patterns
3. **CTDAPD** - Known temporal limitations but validated

### Tier 2: Secondary Investigation Datasets  
4. **NF-ToN-IoT-v3** - NetFlow v3 features, IoT context
5. **NF-CICIDS2018-v3** - Updated CICIDS with NetFlow features
6. **NF-BoT-IoT-v3** - Botnet/IoT hybrid
7. **RT-IoT2022** - Recent IoT dataset

### Tier 3: Specialized Datasets
8. **CTU-13-Dataset** - Botnet traffic (real-world captured)
9. **NSL-KDD** - Legacy but standardized benchmark
10. **RegSOC-KES2021** - Academic research dataset

## 📊 COMPREHENSIVE ANALYSIS FRAMEWORK

### Phase 1: Basic Dataset Characterization
For each dataset, generate:

#### 1.1 Structural Analysis
```python
# Required outputs for each dataset:
dataset_summary = {
    "name": str,
    "file_format": str,  # CSV, Parquet, etc.
    "file_size_mb": float,
    "total_samples": int,
    "feature_count": int,
    "feature_types": {
        "numerical": int,
        "categorical": int, 
        "temporal": int,
        "network_flow": int
    },
    "missing_values": {
        "columns_with_missing": list,
        "total_missing_percentage": float
    }
}
```

#### 1.2 Class Distribution Analysis
```python
class_analysis = {
    "attack_types": dict,  # {attack_type: count}
    "normal_vs_attack_ratio": float,
    "imbalance_severity": str,  # "severe", "moderate", "balanced"
    "smallest_class_percentage": float,
    "largest_class_percentage": float,
    "class_distribution_plot": str  # path to saved plot
}
```

#### 1.3 Temporal Structure Analysis
```python
temporal_analysis = {
    "has_timestamps": bool,
    "timestamp_columns": list,
    "time_span": {
        "start": str,
        "end": str, 
        "duration_days": float
    },
    "temporal_resolution": str,  # "seconds", "minutes", etc.
    "temporal_gaps": list,  # periods with no data
    "attack_temporal_distribution": dict,  # when attacks occur
    "co_occurrence_verified": bool  # CRITICAL: attacks and benign at same times
}
```

### Phase 2: TDA Suitability Assessment

#### 2.1 Network Flow Characteristics
```python
network_analysis = {
    "flow_based_features": list,  # src_ip, dst_ip, port, protocol
    "statistical_features": list,  # packet counts, byte rates
    "temporal_features": list,   # flow duration, inter-arrival times
    "tda_suitable_features": list,  # features suitable for TDA transformation
    "graph_constructible": bool,  # can build network graphs
    "time_series_constructible": bool  # can create temporal sequences
}
```

#### 2.2 Attack Pattern Characteristics
```python
attack_patterns = {
    "attack_types_present": list,
    "apt_like_attacks": list,  # long-duration, subtle attacks
    "burst_attacks": list,     # short-duration, high-intensity
    "persistent_attacks": list, # multi-stage campaigns
    "reconnaissance_patterns": bool,
    "lateral_movement_patterns": bool,
    "exfiltration_patterns": bool
}
```

#### 2.3 Data Leakage Risk Assessment
```python
leakage_risks = {
    "ids_detection_features": list,  # features that leak detection info
    "future_information": list,      # features using future knowledge
    "temporal_ordering_issues": bool,
    "train_test_temporal_overlap": bool,
    "recommended_split_strategy": str,
    "high_risk_features_to_remove": list
}
```

### Phase 3: TDA-Specific Feature Engineering Potential

#### 3.1 Point Cloud Construction Viability
```python
point_cloud_viability = {
    "suitable_embedding_dimensions": list,  # 2D, 3D, high-dim
    "coordinate_features": list,           # what becomes coordinates
    "filtration_parameters": list,        # what drives filtration
    "expected_homology_dimensions": list,  # H0, H1, H2
    "persistence_significance": str        # "high", "medium", "low"
}
```

#### 3.2 Graph-Based TDA Potential  
```python
graph_tda_potential = {
    "node_definition": str,        # IPs, hosts, services
    "edge_definition": str,        # connections, flows, similarities
    "temporal_graph_evolution": bool,
    "multilayer_graph_possible": bool,
    "expected_graph_statistics": dict,
    "clique_complex_viability": str
}
```

#### 3.3 Time Series TDA Potential
```python
time_series_tda = {
    "sliding_window_sizes": list,    # optimal window sizes
    "embedding_dimensions": list,    # delay embedding dimensions  
    "persistence_time_scales": list, # temporal scales for analysis
    "expected_periodic_patterns": bool,
    "anomaly_detection_viability": str
}
```

### Phase 4: Baseline Performance Requirements

#### 4.1 Simple Statistical Baseline
```python
baseline_requirements = {
    "simple_ml_baseline": {
        "method": "Random Forest",
        "expected_f1": float,
        "expected_precision": float,
        "expected_recall": float
    },
    "statistical_baseline": {
        "method": "Statistical Anomaly Detection", 
        "expected_performance": dict
    },
    "tda_performance_threshold": float,  # minimum improvement required
    "computational_budget": str          # acceptable runtime limits
}
```

## 🚨 CRITICAL VALIDATION CHECKPOINTS

### Checkpoint 1: Data Quality Gate
- ✅ Dataset loads without errors
- ✅ Temporal structure is valid
- ✅ Class labels are consistent
- ✅ No obvious data corruption
- ❌ **STOP**: Fix data issues before proceeding

### Checkpoint 2: Temporal Integrity Gate  
- ✅ Attack and benign samples co-occur temporally
- ✅ No temporal data leakage identified
- ✅ Train/test split preserves temporal ordering
- ❌ **STOP**: Cannot proceed with temporally invalid data

### Checkpoint 3: TDA Viability Gate
- ✅ Dataset has features suitable for TDA transformation
- ✅ Expected topological structures are theoretically meaningful
- ✅ Computational requirements are feasible
- ❌ **STOP**: TDA may not be suitable for this dataset

### Checkpoint 4: Baseline Performance Gate
- ✅ Simple baseline achieves reasonable performance
- ✅ TDA has theoretical advantage for this data type
- ✅ Problem complexity justifies TDA overhead
- ❌ **STOP**: Reconsider TDA applicability

## 📁 OUTPUT STRUCTURE

### Analysis Results Location
```
notebooks/
├── dataset_suitability_analysis_plan.md        # This plan
├── dataset_analysis_[dataset_name].ipynb       # Individual analyses
├── dataset_comparison_summary.md               # Cross-dataset comparison
└── dataset_analysis_results/                   # Generated outputs
    ├── [dataset_name]/
    │   ├── structural_analysis.json
    │   ├── temporal_analysis.json  
    │   ├── tda_suitability.json
    │   ├── baseline_requirements.json
    │   └── plots/
    │       ├── class_distribution.png
    │       ├── temporal_coverage.png
    │       ├── feature_correlation.png
    │       └── attack_timeline.png
    └── cross_dataset_comparison.json
```

### Validation Script Requirements
Each analysis must generate:
1. **Reproducible validation script**: `validate_dataset_analysis_[name].py`
2. **Evidence package**: All plots, metrics, and raw outputs
3. **Risk assessment**: Explicit data leakage and quality warnings
4. **TDA readiness score**: Quantitative suitability assessment

## 🎯 SUCCESS CRITERIA

### Technical Success
- All Tier 1 datasets analyzed with complete evidence packages
- Temporal integrity validated for all datasets
- Clear TDA suitability rankings established
- Baseline performance requirements documented

### Strategic Success  
- Dataset selection rationale is evidence-based
- Risk mitigation strategies are documented
- Computational budgets are realistic
- Pivot criteria are clearly defined

### Compliance Success
- All analysis follows UNIFIED_AGENT_INSTRUCTIONS.md requirements
- Every claim is backed by validation scripts
- Failure modes are honestly documented
- Evidence packages are complete and reproducible

## 🚦 EXECUTION PRIORITY

### Week 1: Foundation (Immediate)
1. **UNSW-NB15** - Validate current working dataset
2. **CIC-IDS-2017** - Analyze primary TDA target
3. **CTDAPD** - Understand temporal limitations

### Week 2: Expansion
4. **NF-ToN-IoT-v3** - NetFlow feature analysis
5. **NF-CICIDS2018-v3** - Updated CIC analysis
6. **Cross-dataset comparison** - Establish rankings

### Week 3: Optimization
7. **Remaining Tier 2/3 datasets** - Complete analysis
8. **Final dataset selection** - Evidence-based decision
9. **TDA implementation roadmap** - Dataset-specific strategies

## ⚠️ FAILURE HANDLING

### If Dataset Analysis Reveals Issues
- **Data Quality Problems**: Document and either fix or exclude dataset
- **Temporal Integrity Violations**: Exclude dataset or find alternative split
- **TDA Unsuitability**: Honestly document limitations, consider statistical alternatives
- **Performance Baseline Issues**: Investigate root causes, adjust expectations

### When to Pivot
- >50% of datasets show temporal integrity violations  
- TDA suitability scores are consistently low (<3/5)
- Baseline performance requirements cannot be met
- Computational budgets are exceeded by >5x

**Remember**: Finding datasets unsuitable for TDA is as valuable as finding suitable ones. Honest assessment prevents wasted development effort.
