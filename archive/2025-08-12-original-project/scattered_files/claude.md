# Claude Project Documentation

## 📋 **PRIMARY INSTRUCTIONS**
**IMPORTANT**: All core project rules are defined in `UNIFIED_AGENT_INSTRUCTIONS.md`. Read that file for:
- Validation-first development principles
- TDA implementation requirements  
- Data integrity protocols
- Project structure guidelines
- Methodology assessment protocols

This file contains Claude-specific extensions and workflow details.

## 🚨 CRITICAL ACCURACY AND VALIDATION RULES

**🛑 MANDATORY - NO EXCEPTIONS:**

### 0. **PROJECT FOCUS: CYBERSECURITY ATTACK DETECTION**
   - **PRIMARY GOAL**: Detect network attacks (DDoS, Brute Force, SQL Injection, etc.)
   - **SUCCESS METRIC**: Attack detection F1-scores, NOT overall accuracy
   - **FAILURE DEFINITION**: 0% attack detection = COMPLETE FAILURE regardless of overall accuracy
   - **HONEST REPORTING**: Never present normal traffic classification as attack detection success

### 1. **VALIDATION-FIRST DEVELOPMENT PRINCIPLE**
   - **ACCURACY > PROGRESS**: Accurate reporting is INFINITELY more valuable than artificial progress claims
   - **FAILURE IS PROGRESS**: Finding methods that don't work is as valuable as finding ones that do
   - **VALIDATE IMMEDIATELY**: Every performance claim must be validated with independent reproduction script
   - **NO CLAIMS WITHOUT PROOF**: Zero tolerance for unvalidated performance assertions
   - **ATTACK-FOCUSED METRICS**: Always report attack detection rates first, overall metrics second

### 2. **MANDATORY VALIDATION PROTOCOL**
   ```python
   # Every performance claim must pass this validation
   def validate_performance_claim(claimed_f1, method_script):
       actual_f1 = run_validation_script(method_script, random_state=42)
       tolerance = 0.05  # 5% tolerance for randomness
       
       if abs(actual_f1 - claimed_f1) > tolerance:
           raise ValueError(f"CLAIM INVALID: {claimed_f1:.3f} vs {actual_f1:.3f}")
       
       return True  # Claim validated
   ```

### 3. **EVIDENCE-BASED REPORTING ONLY**
   - **Every metric**: Must include exact reproduction script path
   - **Every claim**: Must show actual test output with confusion matrices
   - **Every result**: Must be deterministic with fixed random seeds
   - **Every documentation**: Must reference validation that confirms the claim

### 4. **COMPREHENSIVE FAILURE DOCUMENTATION**
   - **Report all failures**: Document what didn't work and why
   - **Quantify failures**: Show exact performance gaps vs. expectations
   - **Learn from failures**: Extract actionable insights for future development
   - **Celebrate failures**: Failed experiments prevent wasted effort on bad approaches

### 5. **VALIDATION FILE STRUCTURE**
   ```
   validation/
   ├── validate_hybrid_tda.py        ✅ (70.6% F1 - VALIDATED)
   ├── validate_supervised_tda.py    ❌ (55.6% F1, not 80% - CLAIM REJECTED)  
   ├── validate_multiscale_tda.py    ? (needs validation)
   └── validation_results.json       # All validated claims registry
   ```

### 6. **MANDATORY GRANULAR VALIDATION OUTPUT STRUCTURE** 
   **REQUIRED FOR ALL VALIDATIONS:** Every TDA validation MUST produce granular data outputs:
   ```
   validation/
   ├── method_name_description/           # Descriptive method name
   │   └── YYYYMMDD_HHMMSS/              # Timestamp-based run directory
   │       ├── VALIDATION_REPORT.md      # Main validation report
   │       ├── validation_summary.json   # Structured summary data
   │       ├── data/                     # Dataset info and raw data
   │       │   ├── raw_data.json
   │       │   ├── persistence_diagrams/ # TDA artifacts by attack type
   │       │   │   ├── normal_H0.json
   │       │   │   ├── normal_H1.json
   │       │   │   ├── ddos_H0.json
   │       │   │   ├── ddos_H1.json
   │       │   │   └── [attack_type]_H[dim].json
   │       │   └── barcodes/             # Persistence barcodes
   │       │       ├── normal_barcodes.json
   │       │       └── [attack_type]_barcodes.json
   │       ├── plots/                    # All visualization outputs
   │       │   ├── confusion_matrix.png
   │       │   ├── precision_recall_curve.png
   │       │   ├── roc_curve.png
   │       │   ├── attack_type_performance.png      # F1/precision/recall by attack type
   │       │   ├── attack_type_confusion_matrices.png # Individual matrices per attack
   │       │   ├── persistence_diagrams.png         # H0, H1, H2 diagrams
   │       │   ├── persistence_landscapes.png       # Persistence landscapes
   │       │   └── topological_features_comparison.png # Topology by attack type
   │       └── results/                  # Metrics and console output
   │           ├── metrics.json          # MUST include attack-type breakdown
   │           ├── topological_analysis.json # TDA-specific metrics
   │           ├── console_output.txt
   │           └── validation_summary.json
   ```

   **MANDATORY METRICS IN metrics.json:**
   ```json
   {
     "overall_metrics": {
       "f1_score": float, "accuracy": float, "precision": float, "recall": float, "roc_auc": float
     },
     "attack_type_metrics": {
       "Normal": {"f1": float, "precision": float, "recall": float, "support": int},
       "DDoS": {"f1": float, "precision": float, "recall": float, "support": int},
       "[AttackType]": {"f1": float, "precision": float, "recall": float, "support": int}
     },
     "topological_analysis": {
       "homology_dimensions_analyzed": ["H0", "H1", "H2"],
       "persistence_features_extracted": int,
       "topological_separability_score": float,
       "attack_type_topology_signatures": {
         "Normal": {"H0_features": int, "H1_features": int, "avg_persistence": float},
         "[AttackType]": {"H0_features": int, "H1_features": int, "avg_persistence": float}
       }
     }
   }
   ```

   **MANDATORY TDA ARTIFACTS:**
   - **Persistence Diagrams**: H0, H1, H2 for each attack type
   - **Barcodes**: Persistence intervals for topological features
   - **Topological Signatures**: Attack-specific topological characteristics
   - **Landscape Comparisons**: Statistical summaries of persistence landscapes
   
   **Examples from existing project:**
   - `validation/unsw_nb15_granular_analysis/20250807_013922/` (has attack-type breakdown)
   - `validation/unsw_nb15_clean_streaming/20250807_014939/` (lacks granular analysis - INCOMPLETE)

### 7. **PERFORMANCE CLAIM FORMAT**
   ```
   ❌ WRONG: "Achieved 80% F1-score with TDA + Supervised method"
   ✅ RIGHT: "Achieved 55.6% F1-score (validation: validate_supervised_tda.py, seed=42)"
   
   ❌ WRONG: "Method shows promising results"
   ✅ RIGHT: "Method failed: 18.2% F1 vs 70.6% baseline (-52.4%, abandoning approach)"
   ```

### 8. **🚨 NEVER CREATE GENERIC OR SYNTHETIC VALIDATION TESTS**
   - **ONLY test SPECIFIC named methods**: hybrid_multiscale_graph_tda, implement_multiscale_tda, etc.
   - **NO generic "detector" tests**: Do NOT create tests for generic "APTDetector" classes  
   - **NO synthetic data fallbacks**: If real data fails, FIX the data loading, don't create fake data
   - **NO "comprehensive" or "enhanced" invented methods**: Only test methods explicitly mentioned by user
   - **FIND AND TEST EXISTING SCRIPTS**: Look for and run the actual TDA method scripts the user references
   - **DO NOT INVENT NEW TEST APPROACHES**: Use exactly what the user asks for, nothing else

### 9. **WHEN USER SAYS "TEST METHOD X" - DO THIS:**
   1. Find the existing script for method X (e.g., `hybrid_multiscale_graph_tda.py`) 
   2. Run that EXACT script on real data
   3. Report the results from THAT script
   4. DO NOT create a new "validation" wrapper
   5. DO NOT create generic detector tests
   6. DO NOT invent "comprehensive" approaches

### 10. **DISCREPANCY INVESTIGATION PROTOCOL**
   When validation != claim:
   1. **Immediate Documentation**: Record exact discrepancy
   2. **Root Cause Analysis**: Identify technical causes (features, parameters, etc.)
   3. **Process Analysis**: Identify systemic causes (bias, pressure, etc.)
   4. **Corrective Action**: Update process to prevent similar issues
   5. **Honest Correction**: Update all documentation with validated results

**VIOLATION CONSEQUENCES:** Any unvalidated claims immediately invalidate ALL results and require complete re-validation of project status.

## 🚨 METHODOLOGY FAILURE ASSESSMENT PROTOCOL

**🛑 MANDATORY - EARLY FAILURE DETECTION:**

1. **BASELINE PERFORMANCE REQUIREMENTS**
   - Every new methodology MUST be compared against a simple baseline
   - If new method performs worse than baseline, immediately flag as FAILURE
   - Document what went wrong, don't try to fix complex failures
   - Example baselines: random classifier, simple statistical methods, existing solutions

2. **DEGRADATION DETECTION CRITERIA**
   - Performance drops >5% from baseline: ⚠️ **WARNING** - investigate immediately
   - Performance drops >10% from baseline: ❌ **FAILURE** - stop development, analyze root cause
   - Method produces nonsensical results: ❌ **CRITICAL FAILURE** - abandon approach
   - Computational cost >5x baseline without performance gain: ⚠️ **EFFICIENCY WARNING**

3. **HONEST FAILURE COMMUNICATION**
   - State failures clearly: "Method X failed because Y"
   - Don't euphemize: avoid "needs optimization" when you mean "doesn't work"
   - Quantify the failure: show actual vs expected performance numbers
   - Explain impact: how does this affect project timeline/goals

4. **GO/NO-GO DECISION FRAMEWORK**
   - After 3 failed improvement attempts: STOP and reassess fundamental approach
   - If core methodology shows no promise after proper testing: PIVOT to alternatives
   - Document decision rationale with data
   - Update project priorities based on what actually works

5. **RECOVERY ACTIONS**
   - **Minor Issues (<10% degradation)**: Debug systematically, test incrementally
   - **Major Failures (>10% degradation)**: Abandon approach, document lessons learned
   - **Fundamental Problems**: Reassess core assumptions, consider different methodologies
   - **Complete Pivot**: Redirect effort to working approaches, maintain project momentum

**EXAMPLES OF PROPER FAILURE COMMUNICATION:**
- ❌ WRONG: "The improved detector needs some optimization"
- ✅ RIGHT: "The improved detector failed - 50% accuracy vs 82% baseline. Abandoning this approach."

- ❌ WRONG: "We're making progress on the algorithm"  
- ✅ RIGHT: "Three optimization attempts failed. Performance degraded 15%. Pivoting to different methodology."

**VIOLATION CONSEQUENCES:** Continuing failed approaches without honest assessment wastes time and resources.

## 🚨 Session Startup Protocol

**FIRST ACTION EVERY SESSION:** Always read and update `PROJECT_STATUS.md` to understand current project state, priorities, and progress.

```bash
# Check current project status
cat PROJECT_STATUS.md

# Update status document after completing tasks
# (Update completion percentages, move tasks from pending to in_progress/completed)
```

**Key Session Commands:**
```bash
# Quick environment check
source .venv/bin/activate && python -c "import src; print('✅ Environment ready')"

# Check git status
git status

# Run current examples
python examples/apt_detection_example.py

# Run tests
python -m pytest tests/ -v
```

## Project Overview

This is a **Topological Data Analysis (TDA) Platform** designed for cybersecurity and financial risk applications. The platform leverages advanced mathematical methods for pattern recognition in high-dimensional datasets, targeting two key market opportunities:

### Strategic Focus Areas

1. **Cybersecurity MVP (SME Market)**
   - Advanced Persistent Threat (APT) detection
   - IoT device classification and anomaly detection
   - Network intrusion analysis
   - Target: Small-to-medium enterprises requiring interpretable security solutions

2. **Financial Risk MVP (Mid-Market Institutions)**
   - Cryptocurrency market analysis and bubble detection
   - Multi-asset portfolio risk assessment
   - Market regime identification
   - Target: Mid-market financial institutions needing regulatory-compliant risk tools

## Technical Architecture

### Core TDA Methods (`src/core/`)

- **Persistent Homology** (`persistent_homology.py`): Robust topological feature extraction using ripser/gudhi
- **Mapper Algorithm** (`mapper.py`): Network-based data visualization and analysis
- **Topology Utilities** (`topology_utils.py`): Distance computation, dimension estimation, preprocessing

### Domain Applications

#### Cybersecurity (`src/cybersecurity/`)

- **APT Detection** (`apt_detection.py`): Long-term threat pattern identification with 98%+ accuracy potential
- **IoT Classification** (`iot_classification.py`): Device fingerprinting and spoofing detection
- **Network Analysis** (`network_analysis.py`): Real-time anomaly detection

#### Finance (`src/finance/`)

- **Crypto Analysis** (`crypto_analysis.py`): Bubble detection (60% sensitivity 0-5 days ahead)
- **Risk Assessment** (`risk_assessment.py`): Multi-asset risk aggregation
- **Market Analysis** (`market_analysis.py`): Regime identification and transition detection

### Shared Utilities (`src/utils/`)

- Data preprocessing, visualization, and model evaluation tools

## Key Technical Advantages

1. **Mathematical Interpretability**: Unlike black-box ML models, provides explainable topological features
2. **Noise Robustness**: Persistent homology stable under small perturbations
3. **High-Dimensional Performance**: Superior pattern recognition in complex datasets
4. **Regulatory Compliance**: Explainable AI capabilities meet regulatory requirements

## Development Environment

### Python Environment

- **Virtual Environment**: `.venv/` (Python 3.13.3)
- **Activation**: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
- **Python Path**: `/home/stephen-dorman/dev/TDA_projects/.venv/bin/python`

### Key Dependencies

```text
# Core TDA Libraries
scikit-tda>=1.0.0
gudhi>=3.8.0
ripser>=0.6.0
persim>=0.3.0
kmapper>=2.0.0

# ML/Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.9.0
xgboost>=1.5.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
networkx>=2.6.0

# Domain-Specific
scapy>=2.4.5 (cybersecurity)
yfinance>=0.1.70 (finance)
```

## Code Style Guidelines

### General Conventions

- **PEP 8** style with 88-character line length (Black formatter)
- **Type hints** for all public methods and functions
- **NumPy/SciPy docstring format** for comprehensive documentation
- **Robust error handling** with informative messages
- **Verbose logging** options for debugging complex TDA computations

### TDA-Specific Patterns

- Always check for **minimum 3 points** before TDA computation
- Handle **infinite persistence features** gracefully
- Use **consistent feature naming**: `ph_dim{X}_{metric}`, `mapper_{property}`, `stat_{measure}`
- Implement **fallback mechanisms** when TDA computation fails (return zero features)
- Prefer **stable topological features** over high-resolution unstable ones

### Architecture Patterns

- **sklearn-compatible estimators** (BaseEstimator, TransformerMixin)
- **Core Module**: Fundamental algorithms (persistent homology, mapper, utilities)
- **Domain Modules**: Separate cybersecurity and finance applications inheriting from core
- **Utilities**: Shared preprocessing, visualization, and evaluation tools

## Domain-Specific Knowledge

### Cybersecurity Context

- **APTs**: Advanced Persistent Threats - long-term, stealthy attacks requiring subtle pattern detection
- **IoT**: Internet of Things devices - often vulnerable, diverse protocols, need device fingerprinting
- **Network Analysis**: Focus on traffic patterns, device behaviors, balance sensitivity vs false positives
- **Regulatory**: SEC 4-day incident reporting, EU NIS 2 directive requirements

### Financial Context

- **Market Regimes**: Bull/bear/volatile periods with different topological characteristics
- **Bubble Detection**: Rapid price increases followed by crashes, combine topological with traditional indicators
- **Risk Management**: VaR, correlation analysis, stress testing with mathematical interpretability
- **Regulatory**: DORA compliance, Basel III requirements, focus on explainability and audit trails

## Working Examples

### APT Detection Example (`examples/apt_detection_example.py`)

Successfully demonstrates:

- Synthetic network data generation with embedded APT patterns
- TDA-based feature extraction (persistent homology + mapper)
- 82% overall accuracy, 68% APT recall on test data
- Feature importance analysis
- Long-term temporal threat detection
- Comprehensive visualization and reporting

### Key Performance Metrics Achieved

- **Cybersecurity**: 82% accuracy, 68% recall for APT detection
- **Finance**: HIGH_RISK bubble detection in test scenarios
- **Core TDA**: Successfully processing 50-point circle data, 10x10 distance matrices

## Development Workflow

### 📋 Session Management Protocol

**Every development session MUST:**

1. **Start with STATUS CHECK:**
   ```bash
   cat PROJECT_STATUS.md  # Read current priorities and progress
   git status             # Check for uncommitted changes
   ```

2. **Update PROJECT_STATUS.md** when completing tasks:
   - Change task status: `pending` → `in_progress` → `completed`
   - Update completion percentages in progress dashboard
   - Add new tasks discovered during development
   - Update "Last Updated" date

3. **End session with STATUS UPDATE:**
   ```bash
   # Commit any changes
   git add PROJECT_STATUS.md
   git commit -m "Update project status - [brief description of progress]"
   ```

### Daily Development Routine

```bash
# Morning startup
source .venv/bin/activate
cat PROJECT_STATUS.md
git status

# Work on current priority tasks from PROJECT_STATUS.md
# Update status document as tasks progress

# End of session
git add . && git commit -m "Session progress: [description]"
grep -A 10 "Active Development Priorities" PROJECT_STATUS.md  # Review next steps
```

### Running Examples

```bash
# Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run APT detection example
python examples/apt_detection_example.py

# Run tests
python -m pytest tests/ -v

# Install in development mode
pip install -e .
```

### Essential Development Commands

```bash
# Environment Management
source .venv/bin/activate                    # Activate virtual environment
pip install -e .                            # Install package in development mode
pip install -r requirements.txt             # Install/update dependencies

# Testing & Validation
python -m pytest tests/ -v                  # Run full test suite
python -m pytest tests/test_core.py -v      # Run specific module tests
python -m pytest --cov=src tests/           # Run with coverage report

# Code Quality
python -m black src/ tests/                 # Format code
python -m flake8 src/ tests/                # Check style
python -m mypy src/                         # Type checking

# Quick Module Tests
python -c "from src.core.topology_utils import create_point_cloud_circle; print('✅ Core working!')"
python -c "from src.cybersecurity.apt_detection import APTDetector; print('✅ Cybersecurity working!')"
python -c "from src.finance.crypto_analysis import CryptoAnalyzer; print('✅ Finance working!')"

# Examples & Demos
python examples/apt_detection_example.py    # Run APT detection demo
python -c "import matplotlib.pyplot as plt; plt.ioff()"  # Test visualization setup

# Git Workflow
git status                                  # Check working directory status
git add -A && git commit -m "Description"   # Stage and commit changes
git log --oneline -5                       # View recent commits
git diff                                   # View unstaged changes

# Project Status Management
cat PROJECT_STATUS.md                      # View current project state
grep -E "🔄|❌|✅" PROJECT_STATUS.md      # Quick status overview
```

### Performance & Debugging Commands

```bash
# Memory & Performance Monitoring
python -m memory_profiler examples/apt_detection_example.py  # Memory profiling
python -m cProfile -s tottime examples/apt_detection_example.py  # Performance profiling

# TDA-Specific Debugging
python -c "import ripser; from src.core.persistent_homology import PersistentHomology; print('✅ TDA libraries ready')"
python -c "import gudhi; print(f'GUDHI version: {gudhi.__version__}')"

# Data Validation
python -c "import numpy as np; from src.utils.data_preprocessing import validate_input_data; print('✅ Data validation ready')"

# Visualization Testing
python -c "import matplotlib; matplotlib.use('Agg'); print('✅ Matplotlib headless mode')"
```

### File Structure

```text
TDA_projects/
├── src/
│   ├── core/              # Core TDA algorithms
│   ├── cybersecurity/     # APT detection, IoT classification
│   ├── finance/           # Crypto analysis, risk assessment
│   └── utils/             # Shared utilities
├── examples/              # Working demonstration scripts
├── tests/                 # Unit tests
├── .venv/                 # Python virtual environment
├── .vscode/               # VS Code settings and chat configuration
├── workspace_chats/       # Chat logs and session summaries
├── PROJECT_STATUS.md      # Living project status document (READ FIRST!)
├── claude.md             # This development guide
├── requirements.txt       # Dependencies
├── setup.py              # Package configuration
└── README.md             # Project documentation
```

## Performance Considerations

### TDA Computational Guidelines

- **Data Subsampling**: Use farthest point sampling for large datasets (>1000 points)
- **Distance Matrix**: O(n²) scaling - subsample when n > 500
- **Filtration Thresholds**: Set appropriate thresholds to avoid computational explosion
- **Backend Selection**: Use ripser as default, gudhi as alternative
- **Mapper Parameters**: Default 10-15 intervals with 30-40% overlap

### Error Handling Patterns

- Wrap TDA computations in try-catch blocks
- Provide meaningful fallbacks when topology computation fails
- Validate input data dimensions and types
- Handle edge cases (empty data, single points, high noise)

## Testing Strategy

### Test Data Patterns

- **Synthetic Geometric Data**: Circles, torus, spheres for validation
- **Edge Cases**: Empty data, single points, high noise scenarios
- **Mathematical Properties**: Persistence stability, mapper connectivity
- **Deterministic Seeds**: Reproducible tests with fixed random seeds

### Validation Approaches

- Compare against known topological properties
- Stability analysis under noise perturbations
- Performance benchmarking against baseline methods
- Cross-validation on domain-specific datasets

## Common Development Gotchas

1. **TDA Computation Expense**: Always validate input size before processing
2. **Infinite Persistence Features**: Need special handling in feature extraction
3. **Empty Persistence Diagrams**: Valid outputs requiring zero-feature fallbacks
4. **Time Series Embeddings**: Require sufficient window sizes for meaningful topology
5. **Memory Usage**: Distance matrix computations can exceed available RAM

## Project Management & Workflow

### 🔄 Active Status Tracking

**CRITICAL:** Always maintain `PROJECT_STATUS.md` as the single source of truth for:
- Current phase progress and completion percentages
- Active development priorities and sprint backlog
- Completed milestones and recent accomplishments
- Upcoming dependencies and potential blockers

### 📊 Status Update Commands

```bash
# Quick status overview
grep -E "Phase 1 Progress|Overall Progress" PROJECT_STATUS.md

# View current priorities
grep -A 20 "Active Development Priorities" PROJECT_STATUS.md

# Check recent accomplishments
grep -A 10 "Recent Accomplishments" PROJECT_STATUS.md

# Update last modified date
sed -i "s/Last Updated:.*/Last Updated:** $(date '+%B %d, %Y')/g" PROJECT_STATUS.md
```

### 🎯 Development Phase Tracking

**Current Phase:** MVP Development (Phase 1)  
**Duration:** 6 months  
**Success Criteria:** Production-ready platform with pilot customers

**Key Milestones:**
- [ ] Month 1-2: Infrastructure & Testing (comprehensive test suite, Docker, CI/CD)
- [ ] Month 3-4: Algorithm Optimization (95%+ accuracy, real-time processing)
- [ ] Month 5-6: Pilot Deployment (3-5 customers, monitoring, compliance)

### Future Development Priorities

**Phase 2: Market Validation (6-12 months)**
- Multi-tenant SaaS architecture
- Customer dashboards and reporting
- 5-10 SME cybersecurity customers
- 3-5 mid-market financial customers

**Phase 3: Platform Integration (12-24 months)**
- Unified cyber-financial risk platform
- Enterprise-grade deployment
- Advanced analytics and AI integration
- Regulatory compliance automation

## Research Validation

The platform is built on proven research demonstrating:

- **98.42% accuracy** in IoT device classification
- **60% sensitivity** in financial bubble detection 0-5 days ahead
- **1.2-2.1% improvement** over state-of-the-art forecasting methods
- **Mathematical interpretability** crucial for regulatory compliance

## Contact and Contribution

This platform targets the convergence of regulatory mandates, skills gaps, and TDA's proven superiority in high-dimensional pattern recognition. The strategic focus balances technical feasibility with substantial market opportunities in sectors experiencing unprecedented demand for sophisticated yet accessible risk management tools.

For development questions, refer to the comprehensive docstrings in each module and the working examples in the `examples/` directory.
