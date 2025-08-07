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

### Strict File Organization
- **NO PYTHON SCRIPTS IN ROOT**: The root directory (`TDA_projects/`) is reserved for project-level configuration files (e.g., `README.md`, `requirements.txt`, `setup.py`, `LICENSE`, `Makefile`, `docker-compose.yml`, `.gitignore`, `pytest.ini`, `gemini.md`, `claude.md`, `UNIFIED_AGENT_INSTRUCTIONS.md`, etc.). No Python scripts (`.py` files) should be created or reside directly in the root.
- **`src/`**: Contains the core, production-ready source code of the TDA platform. This includes algorithms, models, utilities, and API implementations. Code here should be modular, well-tested, and reusable.
- **`validation/`**: Dedicated to all validation scripts (`validate_*.py`, `test_*.py` that are part of the validation framework). Each validation experiment should have its own subdirectory following the `method_name_description/YYYYMMDD_HHMMSS/` structure.
- **`notebooks/`**: For exploratory data analysis (`explore_datasets.ipynb`), method prototyping (`method_prototyping.ipynb`), literature reproduction (`literature_reproduction.ipynb`), and any one-off analysis or debugging scripts (`analyze_*.py`, `debug_*.py`). These are not production code but crucial for understanding and development.
- **`examples/`**: Contains small, self-contained scripts demonstrating how to use parts of the `src/` codebase or showcasing specific functionalities. These are meant for quick understanding and demonstration.
- **`scripts/`**: For general-purpose utility scripts that automate tasks (e.g., `deploy.sh`, `download_cicids2017.py`, `preprocess_cicids2017.py`). These are operational scripts, not core logic or validation.
- **`archive/`**: For outdated, deprecated, or failed experimental scripts that are no longer actively used but might be needed for historical reference. Move files here instead of deleting them.

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

## 🚦 DATASET ANALYSIS AND PRE-CODING PROTOCOL

- **MANDATORY**: Before any code is written, agents must perform and document a full dataset analysis.
- **NO EXCEPTIONS**: Coding without prior data analysis is forbidden.
- **DATA INTEGRITY**: Explicitly check for temporal overlap and label balance; document risks of leakage.
- **TRACEABILITY**: Link every experiment to its dataset analysis and validation output.
- **BASELINE ENFORCEMENT**: Compare all results to baseline; stop and review if performance degrades.
- **FAILURE LOGGING**: Document all errors, root causes, and lessons learned.

- **Dataset Analysis Requirements:**
  - Load and summarize dataset (feature types, missing values, temporal structure, class balance)
  - Document anomalies, limitations, or risks (e.g., temporal gaps, label imbalance)
  - Save analysis in a dedicated notebook or markdown file (e.g., `notebooks/dataset_analysis_[name].ipynb`)
  - Explicitly verify temporal overlap and co-occurrence of attack/benign samples
  - Flag any risk of data leakage before proceeding

- **Coding Protocol:**
  - Reference completed dataset analysis before starting any implementation
  - If analysis is missing, halt and request it
  - For every new method, list expected input/output formats and check compatibility with the dataset
  - Use small test runs to validate data loading and feature extraction before full-scale experiments

- **Documentation:**
  - Link each experiment to its dataset analysis and validation results
  - Log all errors and failures with root cause analysis

## 🛡️ ENHANCED VALIDATION AND REPORTING CONTROLS

- **Raw Output Capture**: Every validation run must save and link raw console output/logs for manual inspection. Raw output files must be included in the evidence package and referenced in documentation.

- **Visualization Suite**: Every experiment must generate and save the following visualizations:
  - Confusion matrix heatmap
  - ROC curve with AUC
  - Precision-Recall curve
  - Feature importance plot
  - Distribution of predictions
  - Raw classification report

- **Evidence Package Enforcement**: No performance claim is valid without a complete evidence package containing all required plots, raw output, and the validation script. All results must reference specific validation files and saved outputs.

- **Technical Validation Gates**: All experiment scripts should use decorators, wrappers, or other technical barriers to prevent reporting results without validation. Results cannot be documented or reported unless validation passes and the evidence package is complete.

- **Systematic Auditing**: Implement regular automated audits of all documented claims to ensure reproducibility and evidence compliance. Audit scripts should check that every claim in documentation is linked to a validation script and evidence package, and that results are within tolerance.

- **Multi-Run Reporting**: All key metrics (F1, accuracy, precision, recall) must be reported as mean ± standard deviation over multiple runs (minimum 5 runs) with fixed random seeds.



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
│       │   ├── roc_curve.png         # New: ROC curve
│       │   ├── precision_recall_curve.png # New: Precision-Recall curve
│       │   ├── feature_importance.png # New: Feature importance
│       │   ├── prediction_distribution.png # New: Distribution of predictions
│       │   ├── attack_type_performance.png
│       │   ├── persistence_diagrams.png
│       │   └── topological_features_comparison.png
│       ├── raw_output/               # New: Raw console output from validation run
│       │   └── console_output.log
│       └── results/                  # Metrics and analysis
│           ├── metrics.json          # Attack-type breakdown required
│           ├── topological_analysis.json # TDA-specific metrics
│           └── validation_summary.json
```

### MANDATORY METRICS FORMAT
```json
{
  "overall_metrics": {
    "f1_score": {"mean": float, "std": float}, # New: Include standard deviation
    "accuracy": {"mean": float, "std": float},
    "precision": {"mean": float, "std": float},
    "recall": {"mean": float, "std": float}
  },
  "attack_type_metrics": {
    "Normal": {"f1": {"mean": float, "std": float}, "precision": {"mean": float, "std": float}, "recall": {"mean": float, "std": float}, "support": int},
    "DDoS": {"f1": {"mean": float, "std": float}, "precision": {"mean": float, "std": float}, "recall": {"mean": float, "std": float}, "support": int},
    "[AttackType]": {"f1": {"mean": float, "std": float}, "precision": {"mean": float, "std": float}, "recall": {"mean": float, "std": float}, "support": int}
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
> "Implemented XYZ. Achieved **75.3% ± 0.02 attack detection F1-score** (Validated by: `validation/validate_xyz_unsw.py` with seed=42).
> **Validation Confirmation**:
> ✅ Claimed 75.3% vs Validated 75.3% (exact match)
> ✅ Reproducible with fixed seed=42
> ✅ All mandatory plots generated and saved (confusion matrix, ROC, PR, feature importance, prediction distribution)
> ✅ Raw console output captured and linked."

### For Failed Experiments  
> "Attempted ABC. Attack detection F1: **12.1% ± 0.01**, a 63% regression from baseline. Abandoning approach. Root cause: temporal data sparsity incompatible with method requirements.
> **Validation Confirmation**:
> ✅ Reproducible with fixed seed=42
> ✅ All mandatory plots generated and saved (confusion matrix, ROC, PR, feature importance, prediction distribution)
> ✅ Raw console output captured and linked."

## 🛡️ SYSTEMATIC AUDITING PROTOCOL
- **MANDATORY**: All performance claims documented in reports or project files must be systematically audited for reproducibility and accuracy.
- **AUDIT FREQUENCY**: Audits should be performed regularly, especially before major project milestones or releases.
- **AUDIT PROCESS**:
    1. Identify all performance claims (e.g., F1-scores, accuracy).
    2. Locate the corresponding validation script and specified random seed.
    3. Re-run the validation script with the exact parameters.
    4. Compare the re-generated results with the claimed results.
    5. Verify that all mandatory plots and raw output are generated and match expectations.
    6. Document the audit outcome (pass/fail) and any discrepancies.
- **FAILURE HANDLING**: If an audit fails (e.g., results are not reproducible, plots are missing), the claim must be immediately flagged, investigated, and corrected. No unverified claims are permitted.

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