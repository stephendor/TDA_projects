# Claude Project Documentation

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

### Common Commands

```bash
# Core functionality test
python -c "from src.core.topology_utils import create_point_cloud_circle; print('Core working!')"

# APT detection test
python -c "from src.cybersecurity.apt_detection import APTDetector; print('Cybersecurity working!')"

# Crypto analysis test
python -c "from src.finance.crypto_analysis import CryptoAnalyzer; print('Finance working!')"
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
├── .github/               # Copilot instructions
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

## Future Development Priorities

### Phase 1: MVP Completion (6-12 months)

- [ ] Enhanced IoT device classification accuracy
- [ ] Real-time APT detection optimization
- [ ] Cryptocurrency derivatives analysis
- [ ] Production-ready deployment pipeline

### Phase 2: Market Expansion (12-24 months)

- [ ] Supply chain risk assessment (NIS 2 compliance)
- [ ] ESG risk integration
- [ ] Enhanced visualization and reporting
- [ ] Enterprise API development

### Phase 3: Platform Integration (24-36 months)

- [ ] Unified cyber-financial risk platform
- [ ] Real-time processing capabilities
- [ ] Advanced analytics and AI integration
- [ ] Regulatory compliance automation

## Research Validation

The platform is built on proven research demonstrating:

- **98.42% accuracy** in IoT device classification
- **60% sensitivity** in financial bubble detection 0-5 days ahead
- **1.2-2.1% improvement** over state-of-the-art forecasting methods
- **Mathematical interpretability** crucial for regulatory compliance

## Contact and Contribution

This platform targets the convergence of regulatory mandates, skills gaps, and TDA's proven superiority in high-dimensional pattern recognition. The strategic focus balances technical feasibility with substantial market opportunities in sectors experiencing unprecedented demand for sophisticated yet accessible risk management tools.

For development questions, refer to the comprehensive docstrings in each module and the working examples in the `examples/` directory.
