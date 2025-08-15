---
description: AI rules derived by SpecStory from the project AI interaction history
globs: *
---

---
description: AI rules for GitHub Copilot aligned with our comprehensive TDA platform development
globs: *
---

# TDA Platform Development Rules for GitHub Copilot

## 📋 **PROJECT OVERVIEW**
**IMPORTANT**: This is a **comprehensive Topological Data Analysis (TDA) platform** that applies advanced mathematical topology and deep learning techniques to finance and cybersecurity. This is NOT a simple vector stack project - it's a full enterprise platform.

## 🚨 **CRITICAL CONTEXT MAINTENANCE**

### **Before Making ANY Suggestions or Changes**
- **ALWAYS** verify this is a comprehensive TDA platform, not a simplified project
- **ALWAYS** maintain the full scope: Core TDA + Finance + Cybersecurity + Streaming
- **NEVER** suggest removing modules or simplifying architecture without explicit user request
- **NEVER** assume the project scope has changed to basic TDA only

### **Context Verification Protocol**
1. **Check Project Scope**: Comprehensive TDA platform for finance and cybersecurity
2. **Verify Architecture**: C++23 core + Python API + FastAPI + Kafka + Flink
3. **Confirm Modules**: Core TDA, finance, cybersecurity, streaming infrastructure
4. **Maintain Performance**: Enterprise-grade requirements (<100ms, >10K events/sec)

### **Red Flag Phrases That Require Immediate Context Check**
- "This seems like a simple project"
- "We could simplify this to just..."
- "This looks like basic TDA only"
- "We don't need all these modules"
- "Let's focus on just the core"

## 🏗️ **CURRENT PROJECT STATE**
- **Project**: Comprehensive TDA Platform for Finance and Cybersecurity
- **Architecture**: C++23 core engine + Python API + FastAPI + Kafka + Flink
- **Scope**: Core TDA algorithms, finance module, cybersecurity module, streaming infrastructure
- **Status**: Fresh start with clean C++23 structure, ready for implementation

## 🚫 **FORBIDDEN ACTIONS**
- **DO NOT** suggest this is a simple vector stack project
- **DO NOT** remove finance or cybersecurity modules from the scope
- **DO NOT** simplify the architecture to basic TDA only
- **DO NOT** ignore the streaming infrastructure requirements
- **DO NOT** suggest removing the comprehensive feature set
- **DO NOT** assume scope has been reduced without explicit evidence

## ✅ **REQUIRED UNDERSTANDING**

### **Project Scope (Comprehensive Platform)**
- **Core TDA Engine**: Persistent homology, multiple filtrations, vectorization methods
- **Finance Module**: Market regime detection, portfolio analysis, systemic risk assessment
- **Cybersecurity Module**: Real-time anomaly detection, threat classification, network analysis
- **Streaming Infrastructure**: Kafka + Flink for real-time processing
- **Performance Requirements**: <100ms latency, >10,000 events/second, >20GB datasets

### **Current Implementation Status**
- **Clean C++23 Structure**: Ready for core TDA engine implementation
- **Modern Build System**: CMake with C++23, SIMD, OpenMP optimization
- **Research Integration**: ICLR challenge winners, Geomstats, Giotto-Deep
- **Zero Legacy Code**: Complete fresh start with focused architecture

## 🔧 **DEVELOPMENT GUIDELINES**

### **C++23 Core Engine Development**
- **Use Modern C++23**: Concepts, ranges, coroutines, modules where applicable
- **Performance Focus**: SIMD optimization, OpenMP parallelization, memory pools
- **Mathematical Rigor**: All TDA algorithms must be mathematically correct
- **Clean Architecture**: Modular design with clear separation of concerns

### **Python API and Integration**
- **FastAPI**: Modern, fast Python web framework for API endpoints
- **Pybind11**: Clean C++ to Python bindings for the core engine
- **Streaming Integration**: Kafka producers/consumers, Flink processing
- **Deep Learning**: PyTorch integration for TDA layers and models

### **Finance Module Requirements**
- **Market Regime Detection**: Topological transitions, stability metrics, early warning signals
- **Portfolio Analysis**: Asset relationship mapping, correlation networks, risk metrics
- **Systemic Risk**: Network-based measures, contagion simulation, stress testing

### **Cybersecurity Module Requirements**
- **Real-time Processing**: Network packet streams, sliding windows, anomaly detection
- **Threat Classification**: DDoS, SQL injection, APT detection with topological features
- **Performance**: <100ms latency, high throughput, low false positive rates

## 📚 **REFERENCE ARCHITECTURE**

### **Directory Structure (Current Clean State)**
```
src/
├── cpp/                    # C++23 TDA core engine
│   ├── core/              # Core TDA algorithms
│   ├── vector_stack/      # Vector stack implementation
│   ├── algorithms/        # VR, Alpha, Čech, DTM filtrations
│   └── utils/             # Performance utilities
├── python/                 # Python API and bindings
└── tests/                  # Comprehensive test suite

include/
└── tda/                    # Public C++ headers

docs/                       # Architecture and API documentation
research/                   # Extracted research content
```

### **Technology Stack**
- **Backend**: C++23 (core), Python 3.9+ (API), FastAPI
- **Streaming**: Apache Kafka, Apache Flink
- **Deep Learning**: PyTorch, custom TDA layers
- **Performance**: SIMD, OpenMP, CUDA (optional)
- **Deployment**: Docker, Kubernetes, horizontal scaling

## 🎯 **IMPLEMENTATION PRIORITIES**

### **Phase 1: Core TDA Engine (Tasks 1-2)**
1. **Persistent Homology**: VR, Alpha, Čech, DTM filtrations
2. **Vectorization**: Deterministic vector-stack methods
3. **Performance**: >1M points in <60 seconds
4. **Python Bindings**: Clean pybind11 integration

### **Phase 2: Backend & Streaming (Task 3)**
1. **FastAPI Application**: C++ TDA core integration
2. **Kafka Cluster**: High-throughput event streaming
3. **Flink Integration**: Stream processing validation
4. **API Endpoints**: TDA computation orchestration

### **Phase 3: Advanced Features (Task 5)**
1. **TDA Layers**: Persistence attention, hierarchical clustering, TDA-GNN
2. **PyTorch Integration**: Seamless deep learning compatibility
3. **Benchmarking**: Performance validation against baselines

### **Phase 4: Domain Modules (Tasks 6-7)**
1. **Finance Module**: Market regime detection, portfolio analysis
2. **Cybersecurity Module**: Anomaly detection, threat classification
3. **Real-time Processing**: Streaming data analysis

### **Phase 5: Performance & Scale (Task 8)**
1. **Distributed Processing**: Spark/Flink optimization
2. **GPU Acceleration**: CUDA implementation
3. **Horizontal Scaling**: Kubernetes deployment
4. **Load Testing**: >10,000 events/second validation

## 🚨 **CRITICAL REQUIREMENTS**

### **Performance Standards**
- **Latency**: <100ms for real-time analysis
- **Throughput**: >10,000 events/second
- **Scale**: >20GB datasets, >1M points
- **Memory**: Streaming processing, no crashes

### **Mathematical Correctness**
- **TDA Algorithms**: Mathematically validated implementations
- **Persistence Diagrams**: Correct birth/death time computation
- **Betti Numbers**: Accurate dimension calculations
- **Vectorization**: Deterministic, reproducible results

### **Enterprise Readiness**
- **Security**: Authentication, authorization, auditing
- **Compliance**: Regulatory requirements for finance
- **Scalability**: Horizontal scaling, load balancing
- **Monitoring**: Performance metrics, health checks

## 🔍 **WHEN COPILOT SUGGESTS CHANGES**

### **Accept These Suggestions**
- **C++23 Modernization**: New language features, performance optimizations
- **Architecture Improvements**: Better modularity, cleaner interfaces
- **Performance Enhancements**: SIMD, parallelization, memory optimization
- **Testing Improvements**: Better test coverage, validation frameworks

### **Reject These Suggestions**
- **Scope Reduction**: Removing finance/cybersecurity modules
- **Architecture Simplification**: Removing streaming infrastructure
- **Technology Changes**: Switching from FastAPI to Flask, Kafka to Redis
- **Performance Degradation**: Removing optimization features

## 📖 **RESOURCES FOR COPILOT**

### **Current Project Files**
- **CMakeLists.txt**: Build system and dependencies
- **build.sh**: Build script with optimization flags
- **README.md**: Project overview and features
- **PROJECT_STRUCTURE.md**: Detailed architecture documentation
- **CLEANUP_SUMMARY.md**: Project restructuring summary

### **Task Management**
- **Taskmaster**: All development tasks and priorities
- **PRD**: Complete product requirements and specifications
- **Architecture**: Clean C++23 structure ready for implementation

## 🎉 **SUCCESS CRITERIA**

### **For Copilot Assistance**
- **Maintains Scope**: Keeps finance and cybersecurity modules
- **Preserves Architecture**: Maintains streaming infrastructure
- **Enhances Performance**: Improves C++23 implementation
- **Follows Guidelines**: Adheres to mathematical rigor and enterprise standards
- **Maintains Context**: Never suggests simplifying comprehensive platform

### **Project Success**
- **Comprehensive Platform**: Full TDA platform with all modules
- **Performance Targets**: Meets latency, throughput, and scale requirements
- **Enterprise Ready**: Security, compliance, scalability features
- **Research Impact**: Breakthrough discoveries in peer-reviewed venues

## 🚨 **CONTEXT MAINTENANCE EXCELLENCE**

### **Daily Context Verification**
- **Start of Session**: Always verify comprehensive TDA platform scope
- **Before Major Changes**: Check context before architectural decisions
- **After User Feedback**: Verify understanding matches their intent
- **Before Rule Updates**: Ensure changes align with current scope

### **Context Documentation**
- **Current Scope**: Comprehensive TDA platform for finance and cybersecurity
- **Architecture**: C++23 core + Python API + streaming infrastructure
- **Modules**: Core TDA, finance, cybersecurity, streaming, performance
- **Technology**: Modern C++23, FastAPI, Kafka, Flink, PyTorch
- **Performance**: Enterprise-grade latency, throughput, and scale

---

**Remember**: This is a **comprehensive enterprise TDA platform**, not a simple research project. Maintain the full scope and architecture while implementing with modern C++23 best practices. **NEVER** assume the scope has been simplified without explicit evidence.

## ⚙️ **WORKFLOW & RELEASE RULES**

### General Workflow
- **ALWAYS** activate the virtual environment (`source .venv/bin/activate`) before running any Python scripts or commands.
- **YOU** (the AI) must run all commands, monitor terminal outputs, and handle errors immediately.
- **I** (the user) do not need to be copying or pasting anything at any point, unless explicitly requested.
- **YOU** are forbidden from presenting code in any way other than executable code blocks with Run/Cancel buttons.
- **ALWAYS** perform static analysis (Codacy) after modifying any code.
- **YOU** must activate the zen, sequential thinking and code analysis tools when creating a plan for tackling tasks in the taskmaster MCP system.
- **YOU** must use the existing task 2 subtasks in taskmaster and mark their status as you go, working directly with the PRD and plan already available, without generating additional tasks.

### Experimentation
- For controlled experiments, ensure a deterministic setup: same seed, identical non-filtration flags.
- If point cloud coverage is low (<80%), abort the variant run.
- If Gudhi import fails, classify the variant as SKIPPED.
- Monitor H1 interval counts for over-pruning.
- Verify manifest hash changes appropriately, incorporating filtration mode and parameters.
- If filtration recompute is inactive, generate a warning and optionally fail-fast.

### Success Criteria

- Keep only variants meeting PASS or (for sparse) efficiency PASS criteria.
- Document negative results rigorously—failure is acceptable, unrecorded effort is not.

## 🛠️ **DEBUGGING**

- If a required diagnostics key is missing, patch the code to include it before running any variant runs.
- If any variant run crashes, analyze the terminal output, fix the issue, and rerun the variant.
- If the manifest hash does not change as expected for a variant, investigate the hashing logic and correct it.
- If filtration recompute is not active (attempted_count is 0), ensure that point clouds are generated and accessible in the expected location.

## 🧪 **VALIDATION & TESTING**

- If leakage is detected (potential_leakage_flag is true), stop all further tuning and investigate manifest construction.
- Before accepting a new configuration, always validate full vs pruned pipeline once.

## 🗑️ **METHODOLOGY GRAVEYARD**
- Document all failed approaches and their rationale in `METHODOLOGY_GRAVEYARD.md`.

## 📈 **PERFORMANCE TUNING**

- After representation gains, calibrate the best-performing model only.
- Consider Elastic Net LR (l1_ratio ~0.2–0.4) for structured sparsity.

## 📊 **REPORTING**

- Record results in `daily_logs/20250810_filtration_experiment_plan.md`.

## 💾 **VERSION UPDATES**
- Ensure changes to parameters and settings are reflected in manifest hashes.

## 🧰 **TECH STACK**
- **Gudhi:** (version 3.11.0 or later) is used for filtration computations.
- **nlohmann/json:** (version to be determined) is used for JSON handling.

## 📚 **PROJECT DOCUMENTATION & CONTEXT SYSTEM**
- UNIFIED_AGENT_INSTRUCTIONS.md contains high-level instructions for the AI coding assistant.