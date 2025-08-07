# Unified Agent Instructions for TDA Project

**Purpose:** Ensure consistent behavior across all AI agents (Claude, Copilot, Gemini) working on the TDA cybersecurity platform.

## 🚨 CRITICAL VALIDATION AND ACCURACY RULES

### PRIMARY PROJECT FOCUS: CYBERSECURITY ATTACK DETECTION
- **PRIMARY GOAL**: Detect network attacks (DDoS, Brute Force, SQL Injection, etc.)
- **SUCCESS METRIC**: Attack detection F1-scores, NOT overall accuracy  
- **FAILURE DEFINITION**: 0% attack detection = COMPLETE FAILURE regardless of overall accuracy
- **HONEST REPORTING**: Never present normal traffic classification as attack detection success

### VALIDATION-FIRST DEVELOPMENT PRINCIPLE
- **ACCURACY > PROGRESS**: Accurate reporting is INFINITELY more valuable than artificial progress claims
- **FAILURE IS PROGRESS**: Finding methods that don't work is as valuable as finding ones that do
- **VALIDATE IMMEDIATELY**: Every performance claim must be validated with independent reproduction script
- **NO CLAIMS WITHOUT PROOF**: Zero tolerance for unvalidated performance assertions
- **ATTACK-FOCUSED METRICS**: Always report attack detection rates first, overall metrics second

### MANDATORY VALIDATION PROTOCOL
```python
# Every performance claim must pass this validation
def validate_performance_claim(claimed_f1, method_script):
    actual_f1 = run_validation_script(method_script, random_state=42)
    tolerance = 0.05  # 5% tolerance for randomness
    
    if abs(actual_f1 - claimed_f1) > tolerance:
        raise ValueError(f"CLAIM INVALID: {claimed_f1:.3f} vs {actual_f1:.3f}")
    
    return True  # Claim validated
```

## 🔬 MANDATORY TDA IMPLEMENTATION RULES

### REAL TOPOLOGICAL ANALYSIS REQUIRED
- **THIS IS A TOPOLOGICAL DATA ANALYSIS PROJECT**: ALL analysis MUST use actual topology
- **FORBIDDEN**: Using statistical features (mean, std, skew, kurtosis) as "topological proxies"
- **REQUIRED**: Must use existing TDA infrastructure:
  - `src.core.persistent_homology.PersistentHomologyAnalyzer` 
  - `src.core.mapper.MapperAnalyzer`
  - Real persistence diagrams, birth/death times, Betti numbers
- **FORBIDDEN**: Creating custom "extract_basic_topo_features" or similar statistical proxy functions

### PRE-IMPLEMENTATION VERIFICATION
Before writing ANY TDA validation code, must provide:
1. Exact import statements from existing TDA infrastructure
2. Specific methods that will be called (e.g., `.fit()`, `.transform()`)
3. What actual topological features will be extracted
4. Proof that data split prevents temporal leakage

## 📊 DATA INTEGRITY AND LEAKAGE PREVENTION

### TEMPORAL VALIDATION REQUIREMENTS
- **FORBIDDEN**: Claiming "cross-temporal validation" without proving temporal integrity
- **REQUIRED**: Must verify actual temporal overlap between attack and benign samples
- **FORBIDDEN**: Using temporally separated data (e.g., Feb 14 benign vs Feb 28 attacks)
- **REQUIRED**: Show timestamp analysis proving co-occurring samples

### DATA LEAKAGE DETECTION
- Remove IDS detection outputs (IDS_Alert_Count, Anomaly_Score, etc.)
- Use only clean network flow features
- Validate no future information leaks into training

## 📁 PROJECT STRUCTURE AND ORGANIZATION

### Directory Structure
```
TDA_projects/
├── src/                          # Core platform (main development)
├── examples/                     # Working examples
├── validation/                   # Validation results and scripts
├── daily_logs/                   # Session logs and progress tracking
├── external_repos/               # External reference implementations
│   ├── tda_cybersecurity_papers/ # Paper implementations  
│   ├── gudhi_examples/           # GUDHI library examples
│   ├── scikit_tda_demos/         # Scikit-TDA demonstrations
│   └── README_external.md        # External repo documentation
├── notebooks/                    # Learning and experimentation
│   ├── explore_datasets.ipynb    # Dataset analysis
│   ├── method_prototyping.ipynb  # Quick experiments
│   └── literature_reproduction.ipynb # Paper reproduction
└── data/                         # Dataset organization
    ├── unsw_nb15/               # UNSW-NB15 dataset (preferred for TDA)
    ├── ctdapd/                  # CTDAPD (poor temporal structure)
    └── dataset_comparison.md    # Dataset suitability analysis
```

### Repository Context Guidelines
- **Main work**: Always in `TDA_projects/` (our platform)
- **Reference code**: Clearly marked in `external_repos/` with README
- **Learning space**: `notebooks/` for experimentation
- **Production validation**: Continue using `validation/` structure

## 🎯 METHODOLOGY AND FAILURE ASSESSMENT

### BASELINE PERFORMANCE REQUIREMENTS
- Every new methodology MUST be compared against a simple baseline
- If new method performs worse than baseline, immediately flag as FAILURE
- Document what went wrong, don't try to fix complex failures

### DEGRADATION DETECTION CRITERIA
- Performance drops >5% from baseline: ⚠️ **WARNING** - investigate immediately
- Performance drops >10% from baseline: ❌ **FAILURE** - stop development, analyze root cause
- Method produces nonsensical results: ❌ **CRITICAL FAILURE** - abandon approach

### HONEST FAILURE COMMUNICATION
- State failures clearly: "Method X failed because Y"
- Don't euphemize: avoid "needs optimization" when you mean "doesn't work"
- Quantify the failure: show actual vs expected performance numbers
- Explain impact: how does this affect project timeline/goals

## 📝 VALIDATION OUTPUT REQUIREMENTS

### MANDATORY GRANULAR VALIDATION STRUCTURE
```
validation/
├── method_name_description/           # Descriptive method name
│   └── YYYYMMDD_HHMMSS/              # Timestamp-based run directory
│       ├── VALIDATION_REPORT.md      # Main validation report
│       ├── data/                     # Dataset info and TDA artifacts
│       │   ├── persistence_diagrams/ # TDA artifacts by attack type
│       │   │   ├── normal_H0.json, normal_H1.json
│       │   │   ├── ddos_H0.json, ddos_H1.json
│       │   │   └── [attack_type]_H[dim].json
│       │   └── barcodes/             # Persistence barcodes
│       ├── plots/                    # All visualization outputs
│       │   ├── confusion_matrix.png
│       │   ├── attack_type_performance.png
│       │   ├── persistence_diagrams.png
│       │   └── topological_features_comparison.png
│       └── results/                  # Metrics and analysis
│           ├── metrics.json          # Attack-type breakdown required
│           ├── topological_analysis.json # TDA-specific metrics
│           └── validation_summary.json
```

### MANDATORY METRICS FORMAT
```json
{
  "overall_metrics": {
    "f1_score": float, "accuracy": float, "precision": float, "recall": float
  },
  "attack_type_metrics": {
    "Normal": {"f1": float, "precision": float, "recall": float, "support": int},
    "DDoS": {"f1": float, "precision": float, "recall": float, "support": int},
    "[AttackType]": {"f1": float, "precision": float, "recall": float, "support": int}
  },
  "topological_analysis": {
    "homology_dimensions_analyzed": ["H0", "H1", "H2"],
    "persistence_features_extracted": int,
    "topological_separability_score": float
  }
}
```

## 🔄 DEVELOPMENT WORKFLOW

### Essential Workflow Steps
1. **Status Check**: Read current priorities from project status files
2. **Implementation**: Write/modify code with TDA focus
3. **Validation**: Create/run deterministic validation scripts
4. **Evidence**: Document exact performance with script references
5. **Update**: Maintain project status and progress tracking

### Essential Commands
```bash
# Environment
source .venv/bin/activate
pip install -e .

# Testing & Validation
python validation/validate_[method]_[dataset].py
pytest tests/ --cov=src

# Code Quality
black src/ tests/ examples/
```

## 📋 REPORTING FORMAT

### For Successful, Validated Changes
> "Implemented XYZ. Achieved **75.3% attack detection F1-score** (Validated by: `validation/validate_xyz_unsw.py` with seed=42)."

### For Failed Experiments  
> "Attempted ABC. Attack detection F1: **12.1%**, a 63% regression from baseline. Abandoning approach. Root cause: temporal data sparsity incompatible with method requirements."

## 🎯 STRATEGIC CONTEXT

- **Primary Goal**: TDA platform for Cybersecurity (APT detection, network anomalies)
- **Secondary Goal**: Financial Risk applications (bubble detection, portfolio analysis)
- **Core Advantage**: Mathematical interpretability + regulatory compliance
- **Current Baseline**: 5.0% attack recall, 100% precision (F1=0.096) with topological dissimilarity
- **Key Datasets**: UNSW-NB15 (preferred), CTDAPD (temporal limitations identified)

## 📚 EXTERNAL LEARNING INTEGRATION

### Reference Implementation Usage
- **External repos**: For learning patterns and proven approaches
- **Adaptation**: Create new methods in `src/` inspired by external patterns  
- **Comparison**: Validate our implementations against established benchmarks
- **Documentation**: Reference external sources in validation reports

### Guidelines
- ✅ Study external implementations for patterns
- ✅ Adapt proven techniques to our codebase
- ✅ Compare results against literature benchmarks  
- ❌ Direct copy/paste without understanding
- ❌ Mix external code into our `src/` directory
- ❌ Modify external repositories (treat as read-only)

---

**Last Updated**: 2025-08-07  
**Applies to**: Claude, Copilot, Gemini, and all AI agents working on TDA project  
**Status**: Primary instruction source - supersedes individual agent files for common principles