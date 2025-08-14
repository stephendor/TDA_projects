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
- **Performance Requirements**: <100ms latency, >10000 events/second, >20GB datasets

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
- **Gudhi:** (version 3.11.0 or later) is used for filtration computations.

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
- When addressing memory blowup issues (reference `docs/troubleshooting/memory_blowup_RCA_steps.md`), prioritize steady but significant improvements, maintaining control for easy reversion and repair. Ensure alignment with Taskmaster planning and Agent Instruction files. Employ a rigorous, scalable approach with extensive testing.
- **Workflow for Executing Plans**: Locate relevant plans (e.g., RCA-related docs and strategy plans), extract actionable steps, and execute them. Begin with builds/tests and documentation updates. Open troubleshooting plans to extract checklists and execute concrete steps (build/validate, monitoring hooks, etc.).
- **Execution Steps for RCA and Strategy Plans**: Read `CMakeLists.txt` and tests CMake to understand how tests are built and where to add new targets, plus verify any sanitizer/-fPIC issues mentioned in the plan. Open the ST-101 memory-optimized test and related tests to see what’s already implemented and identify missing pieces (like MemoryMonitor or StreamingDistanceMatrix) we need to add next. Search the codebase for the MemoryMonitor and StreamingDistanceMatrix classes referenced in the plans/tests to see if they exist, and identify missing files we need to implement.
- **Plan References**: When addressing memory blowup issues, the primary reference documents are `memory_blowup_RCA_steps.md` and `RCA_Plan.md`.
- **Coding and Testing**: When implementing new components like `StreamingDistanceMatrix`, create a reusable component (header + source), wire it into the build, and add a unit test to validate correctness and memory-bounded behavior.

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
- If the process appears to stall, particularly with large datasets, inspect the performance harness source and target wiring to understand the exact parameters, modes, and potential blocking points. Run small DM-only probes and constrained runs with time/memory stats to diagnose the issue.

## 🧪 **VALIDATION & TESTING**

- If leakage is detected (potential_leakage_flag is true), stop all further tuning and investigate manifest construction.
- Before accepting a new configuration, always validate full vs pruned pipeline once.
- Update testing to include a lean run for sanity checking, then a full run. Determine if settings need tuning. Find a lean lower bound and then step up in stages to a full run.
- **ALWAYS** run a lean 200K sanity probe (e.g., maxDim=1, K=8–16, tighter radius or time/pair caps), then step up to fuller config and compare adjacency histograms and peak RSS at each step.
- Validate that DM knn_cap_per_vertex_soft reduces work without unacceptable loss; compare simplex counts and adjacency degree distributions across runs.

## 🗑️ **METHODOLOGY GRAVEYARD**
- Document all failed approaches and their rationale in `METHODOLOGY_GRAVEYARD.md`.

## 📈 **PERFORMANCE TUNING**

- After representation gains, calibrate the best-performing model only.
- Consider Elastic Net LR (l1_ratio ~0.2–0.4) for structured sparsity.

## 📊 **REPORTING**

- Record results in `daily_logs/20250810_filtration_experiment_plan.md`.

## 💾 **VERSION UPDATES**
- Ensure changes to parameters and settings are reflected in manifest hashes.

## 📚 **PROJECT DOCUMENTATION & CONTEXT SYSTEM**
- UNIFIED_AGENT_INSTRUCTIONS.md contains high-level instructions for the AI coding assistant.
- `RCA_Plan.md` contains plans for addressing memory blowup issues.
- `memory_blowup_RCA_steps.md` contains steps for addressing memory blowup issues.
- `docs/performance/README.md`: Contains information on baseline comparisons without memory overlap, including the use of flags such as `--baseline-separate-process` and `--baseline-json-out`.
  - Added a “Baseline comparisons without memory overlap” section with flags and an example.
  - Appended the separate-process baseline JSONL note.

## 🧠 **MEMORY MANAGEMENT STRATEGIES**

### **General Approach**
- Prioritize steady but significant improvements when addressing memory blowup issues (reference `docs/troubleshooting/memory_blowup_RCA_steps.md`).
- Maintain control for easy reversion and repair.
- Ensure alignment with Taskmaster planning and Agent Instruction files.
- Employ a rigorous, scalable approach with extensive testing.

### **Strategic Options**

#### **Option A: Incremental Memory Optimization**
- **Complexity**: ⭐⭐⭐ (Medium)
- **Performance Gain**: 70-80% improvement
- **Risk**: Low
- **Timeline**: 1-2 weeks
- **Approach**:
    1. Fix immediate build issues (CMakeLists.txt)
    2. Implement streaming for distance matrix computation
    3. Add memory pooling for simplex objects
    4. Progressive testing at each stage
- **Pros**:
    - Easy rollback at each stage
    - Minimal disruption to existing code
    - Quick wins possible
- **Cons**:
    - May not achieve full ST-101 compliance
    - Doesn't address fundamental O(n³) complexity

#### **Option B: Algorithmic Overhaul**
- **Complexity**: ⭐⭐⭐⭐⭐ (Very High)
- **Performance Gain**: 95%+ improvement
- **Risk**: High
- **Timeline**: 3-4 weeks
- **Approach**:
    1. Replace Čech complex with approximate algorithms
    2. Implement true streaming TDA pipeline
    3. Add GPU acceleration
    4. Distributed processing framework
- **Pros**:
    - Guaranteed ST-101 compliance
    - Future-proof for even larger datasets
    - Best long-term solution
- **Cons**:
    - High risk of breaking existing functionality
    - Complex mathematical validation required
    - Longer implementation time

#### **Option C: Hybrid Progressive Strategy** ⭐ **RECOMMENDED**
- **Complexity**: ⭐⭐⭐⭐ (Medium-High)
- **Performance Gain**: 85-95% improvement
- **Risk**: Medium (mitigated by stages)
- **Timeline**: 2-3 weeks
- **Approach**:
```markdown
Phase 1 (Days 1-3): Foundation & Quick Wins
├── Fix build system issues
├── Implement basic memory monitoring
├── Add streaming distance computation
└── Validate with 100K points

Phase 2 (Days 4-7): Core Optimizations
├── Implement memory pooling
├── Add block-based processing
├── Optimize data structures (SoA)
└── Validate with 500K points

Phase 3 (Days 8-14): Advanced Features
├── Sparse approximations
├── Landmark selection
├── Optional: GPU acceleration
└── Validate with 1M+ points
```

### **Critical Risks & Mitigations**

#### **Mathematical Correctness Risk**
- **Risk**: Approximation algorithms may lose topological features
- **Mitigation**:
    - Implement quality metrics comparing exact vs approximate results
    - Use configurable approximation thresholds
    - Keep exact algorithm as fallback option

#### **Memory Fragmentation Risk**
- **Risk**: Memory pooling could cause fragmentation issues
- **Mitigation**:
    - Use fixed-size pools for common simplex sizes
    - Implement periodic pool defragmentation
    - Monitor fragmentation metrics

#### **Performance Regression Risk**
- **Risk**: Changes might slow down small dataset processing
- **Mitigation**:
    - Use adaptive algorithms based on dataset size
    - Maintain separate code paths for small vs large datasets
    - Continuous benchmarking with your build matrix system

### **Finance Datasets Characteristics**
- **Tick Data**: ~100-500 features (price, volume, bid/ask spreads, order book depth)
- **Dimensionality**: 10-100 after feature engineering
- **Size**: 1-10M events/day per instrument
- **Distribution**: Heavy-tailed, non-stationary, clustered volatility

### **Cybersecurity/SIEM Datasets Characteristics**
- **Network Traffic**: 20-50 features (IPs, ports, protocols, packet sizes)
- **Log Events**: 50-200 features after parsing
- **Size**: 10M-1B events/day for medium enterprises
- **Distribution**: Highly skewed, bursty, temporal patterns

### **Accuracy Requirements**
- **Target**: **≥98% accuracy** with approximations, 100% for exact algorithms

### **Hardware Optimization Strategy**
#### **Development Environment**
- **CPU**: ~8 cores/16 threads → Optimize for 8-16 parallel workers
- **RAM**: 32GB → Target 24GB max usage (leaving 8GB for OS)
- **GPU**: RTX 3070 (8GB VRAM) → CUDA acceleration for distance matrices

#### **Staging Strategy**
````bash
# Local VM approach (recommended initially)
# filepath: scripts/setup_staging_vm.sh
#!/bin/bash

# Create lightweight Docker-based staging environment
docker run -d \
  --name tda-staging \
  --memory="16g" \
  --cpus="4" \
  -v $(pwd):/workspace \
  ubuntu:22.04

# This simulates production constraints
````

## 🗺️ **IMPLEMENTATION ROADMAP**

### **Phase 1 Enhancements (Memory Optimization)**

- **Implement `StreamingDistanceMatrix`**: Create a batch-driven, bounded memory window for optional sparse edge and kNN thresholding. Integrate with the existing `ParallelDistanceMatrix` where appropriate and expose in `tda::utils`.
- **Add Memory Hooks**: Wire `MemoryMonitor` snapshots into distance computation and reporting for quick telemetry and plan compliance checks.
- **Develop Minimal Tests**: Unit tests for streaming distance on synthetic data, asserting memory growth plateaus and thresholded correctness against dense baselines on small datasets.

### **Phase 2 Enhancements (Core Optimization)**

- **SimplexPool Allocator**: Implement fixed-size pools for common simplex arities, tracking fragmentation metrics.
- **Streaming Čech/VR Variant**: Develop a block-based filtration generation with backpressure on memory, including assertions to prevent OOM.

### **Profiling & Validation**

- **Add Profiler**: Add optional profiler integration and a small harness: `run_performance_validation.sh`, `analyze_performance_logs.py`. Track throughput, peak RSS, H1 counts, and manifest hashing as per plan.

### **Phase 3 Enhancements (SimplexPool and Unit Tests)**
- Before implementing the SimplexPool allocator, determine the correct location for it within the project structure (include/src folders).
- Prior to implementation, consider existing simplex data structures to ensure alignment and consistency in design.
- Create a unit test for the SimplexPool allocator to validate its functionality and memory management.

## 📝 **COMMIT MESSAGE GUIDELINES**

- Ensure commit messages are concise and clearly describe the changes made.
- Include relevant references to plans (e.g., `docs/troubleshooting/RCA_Plan.md`, `memory_blowup_RCA_steps.md`).
- Follow a consistent format (e.g., `feat(utils): add MemoryMonitor telemetry to StreamingDistanceMatrix`).

## 🚨 **TROUBLESHOOTING**

- **Stalling Issues**: If the process appears to stall, particularly with large datasets, inspect the performance harness source and target wiring to understand the exact parameters, modes, and potential blocking points. Run small DM-only probes and constrained runs with time/memory stats to diagnose the issue.

## 📈 **SCALING**

- **Neighbor Accumulation**: For large datasets (e.g., n=200k), limit neighbor accumulation in streaming Čech to K nearest neighbors on the fly to avoid memory spikes due to massive temporary adjacency growth.

- To diagnose 200k run behavior, inspect the perf harness source and the large target wiring to see exact parameters, modes, and potential blocking points. Run a small DM-only probe and a constrained 200k run with time/memory stats.

## ✅ **CODE IMPROVEMENTS**

- Fix the `isValid()` warning in `simplex.cpp` by removing the impossible `< 0` check for unsigned types.

## 🚨 **RACE CONDITION MITIGATION**
- When using `--soft-knn-cap > 0`, **ALWAYS** disable `--parallel-threshold 0` for stable, race-free telemetry, **unless** degree_softcap updates are made thread-safe (e.g., using `std::atomic<uint32_t>` or per-thread local caps with bounded checks).

## ⚖️ **PERFORMANCE VS. ACCURACY TRADEOFFS**

### **Per-Thread Local Counters with Bounded Merges vs. Global Atomics**

- **Performance and Contention**
    - **Per-thread local + bounded merges**
        - Pro: Greatly reduces cache-line bouncing and atomic contention; higher throughput at high core counts and "hot" vertices.
        - Con: Merge phases add small latency spikes; tuning batch size/merge frequency is required.
    - **Global atomics per edge (current parallel path)**
        - Pro: Simple and always correct wrt local increments; no merge phase.
        - Con: Heavy contention on popular vertices; can be 10–50x slower in dense regions (as observed).

- **Memory Footprint**
    - **Per-thread local**
        - Pro: If scoped to block-local vertex ranges (i_len + j_len), overhead is modest and cache-friendly.
        - Con: Full per-thread arrays of size n are infeasible; must use block-local arrays or sparse maps. Sparse maps add hashing overhead.
    - **Global atomics**
        - Pro: Single global array only.
        - Con: False sharing and cache thrash under contention.

- **Correctness vs “soft cap” Semantics**
    - **Per-thread local**
        - Pro: Can bound overshoot tightly (e.g., ≤ per-thread batch size per vertex) with pre-checks against a snapshot.
        - Con: Not strictly exact; edges emitted between snapshot and merge can push degrees slightly above K. Avoiding overshoot exactly requires serializing updates or buffering edges until commit (complex and memory-costly).
    - **Global atomics**
        - Pro: Stronger immediate consistency; each increment is visible. Still may allow minimal overshoot if checks and increments aren’t atomic together, but typically tighter than local-batch.
        - Con: The price is contention.

- **Determinism and Reproducibility**
    - **Per-thread local**
        - Con: Edge emission order varies; degree overshoot and emitted set can differ across runs/cores. Not ideal for canonical telemetry.
    - **Global atomics**
        - Con: Still non-deterministic ordering; typically smaller variance than local-batch, but not guaranteed deterministic.
    - **Race-free sequential threshold (baseline)**
        - Pro: Deterministic and telemetry-stable; preferred for “official” measurements.

- **Implementation Complexity**
    - **Per-thread local**
        - Con: More code paths: thread-local buffers, periodic merges, and careful cap checks. If dropping edges on cap is required, edges must be delayed or retracted (hard).
    - **Global atomics**
        - Pro: Simple to reason about and maintain.

- **Telemetry and SLO Alignment**
    - **Per-thread local**
        - Con: Needs explicit “overshoot budget” metrics (e.g., max/avg overshoot per vertex) to monitor approximation.
    - **Global atomics**
        - Pro: Easier to interpret degree stats; no extra metrics needed beyond existing counters.

- **Pragmatic Guidance**
    - For correctness/telemetry runs and regressions: use race-free threshold path (parallel-threshold=0) when soft-knn-cap > 0.
    - For high-throughput parallel runs:
        - If soft-knn-cap is active and contention hurts latency, per-thread local with bounded merges is recommended.
        - Keep a small per-thread batch (or per-block local arrays) and merge frequently to bound overshoot tightly (e.g., ≤1–2 per vertex).
        - Expose and log an “overshoot” metric and the effective average degree to ensure the cap’s usefulness.
    - Keep the global-atomic path available for simplicity but expect severe slowdowns under dense workloads.

## ⚙️ **CONFIGURATION AND SAFEGUARDS**

- Softcap Local Merge: Default `softcap_local_merge` is off for production. Mark it as experimental in documentation.

## 🛠️ **PARALLEL OVERSHOOT REDUCTION**

- To reduce parallel overshoot, consider mid-block partial merges or smaller tiles; or periodic resnapshot to bound `overshoot_max` tightly.

## ✅ **ACCURACY VALIDATION**

- Perform accuracy checks against small-n exact baselines and adjacency histogram exports for QA.

## 🗺️ **DETAILED IMPLEMENTATION PLAN**

### Phase 1 — CI gating and profiler wiring (fast wins)

- Add profiler harness and scripts (referenced in RCA_Plan.md)
  - Tasks:
    - Add `scripts/run_performance_validation.sh` to orchestrate small/medium probes with flags (serial threshold, soft-cap).
    - Add `scripts/analyze_performance_logs.py` to parse JSONL/CSV (dm_edges, simplices, overshoot, peakMB, adjacency histograms).
    - Document usage in `README.md`.
  - DoD:
    - Running the script produces a summary of overshoot=0 (serial), adjacency degree stats, peakMB, plus a pass/fail footer.

- CI parsing and gating of uploaded artifacts (RCA_Plan.md; memory_blowup_RCA_steps.md)
  - Tasks:
    - Add a CI step to parse JSONL/CSV artifacts from `/tmp/tda-artifacts`.
    - Gate on:
      - serial threshold → overshoot_sum == 0 and overshoot_max == 0.
      - dm_peakMB and rss_peakMB presence with numeric values.
      - adj histogram CSVs exist and non-empty; compute mean/median degree.
    - Persist a baseline artifact and compare new run against last-accepted baseline within tolerance (e.g., ±10% dm_edges/simplices for identical params).
  - DoD:
    - CI fails if overshoot > 0 in serial mode or artifacts are missing/corrupt; logs show parsed metrics and tolerances.

### Phase 2 — Scale-up staircase runs and reporting

- Staircase probes: 200K → 500K → 1M (memory_blowup_RCA_steps.md)
  - Tasks:
    - Add jobs or parametrized script invocations for 200K/500K/1M DM-only and Čech (maxDim=1 for time bounds).
    - Capture time/memory stats; export JSONL + adj hist CSV per run.
    - Write results to `daily_logs/YYYYMMDD_*` (as per project logging practice).
  - DoD:
    - Each size produces artifacts and a summarized log including dm_blocks, dm_edges, simplices, overshoot, peakMB, wall-clock.
    - CI stores them as artifacts; logs include a short comparison table vs. previous acceptable baseline.

- Accuracy footprint with soft-cap vs exact small-n (RCA_Plan.md)
  - Tasks:
    - Define small-n exact runs (e.g., n=2K–4K) using dense baseline and serial threshold.
    - Run soft-cap variants (K=8/16/32) and measure deltas in:
      - edge counts, adjacency degree distributions,
      - H1 interval counts (requires computing persistence diagrams).
    - Emit a compact “accuracy report” JSON (delta_% per metric).
  - DoD:
    - Report JSON attached to CI artifacts; CI warns (but does not fail) if deltas exceed configurable thresholds; documentation updated with guidance.

### Phase 3 — Streaming VR variant with memory backpressure

- Implement block-based VR generation (RCA_Plan.md Phase 2)
  - Tasks:
    - Add a streaming VR builder mirroring Čech’s block/tiling approach with bounded-degree adjacency and memory backpressure.
    - Telemetry parity with Čech: simplex counts by dim, adj histogram, peak memory.
    - CLI flags in the harness to select VR mode; export same JSONL/CSV.
  - DoD:
    - Unit tests on small-n exact parity; perf probe runs produce VR artifacts; CI job exercises VR path and uploads outputs.

### Phase 4 — SimplexPool fragmentation telemetry and maintenance

- Fragmentation metrics and defragmentation hooks (RCA_Plan.md)
  - Tasks:
    - Extend `SimplexPool` to track:
      - allocated pages, free-list size, fragmentation ratio.
    - Add optional defragmentation routine (manual/periodic).
    - Telemetry integration into JSONL (pool_* fields).
  - DoD:
    - Unit tests assert stable metrics under churn scenarios; JSONL shows pool telemetry; docs updated with “when to defragment”.

### Phase 5 — Stalling diagnostics and reliability guards

- Stalling/timeouts with parameter echo (memory_blowup_RCA_steps.md)
  - Tasks:
    - Add harness timeout guard for probes; on timeout, print current params, block indices, and partial telemetry.
    - Add DM-only quick probe in CI with tight timeout; fail-fast with actionable log.
  - DoD:
    - CI shows clear “timeout diagnostics” section when exceeded; partial artifacts still uploaded.

- Manifest hashing and recompute safeguards (memory_blowup_RCA_steps.md)
  - Tasks:
    - Ensure manifest hash includes filtration mode and all params (e.g., radius, K, caps, parallel-threshold, maxDim).
    - Warn/fail if recompute inactive (attempted_count == 0) given a new hash.
    - Add unit tests for hashing determinism and param coverage.
  - DoD:
    - Tests green; CI logs include manifest hash; missing recompute triggers warning/failure per configuration.

### Phase 6 — Optional accelerators and distribution (roadmap alignment)

- GPU acceleration for distance blocks (RCA_Plan.md Phase 3/5)
  - Tasks:
    - Optional CUDA path for blockwise distance; env/flag to enable.
    - Validate correctness on small-n and measure throughput gains.
  - DoD:
    - Feature flag documented; small benchmark shows meaningful speedup; CPU fallback remains default.

- Distributed hooks (Flink/Spark) for batched adjacency (RCA_Plan.md Phase 5)
  - Tasks:
    - Define an export format for block tasks and edge emissions.
    - Provide a minimal adapter or doc example to map blocks into stream processing.
  - DoD:
    - Reference pipeline documented; not gated in CI.

### Cross-cutting acceptance criteria

- Telemetry
  - All runs export JSONL with dm_blocks, dm_edges, simplices, dm_peakMB/rss_peakMB, overshoot_sum/max; adjacency CSVs present when requested.

- Determinism and policy
  - When soft-cap active and accuracy mode selected: `parallel-threshold=0`; overshoot must be 0.

- CI reliability
  - Artifact uploads are stable; parser reports a succinct summary and enforces critical gates.

## COMMIT MESSAGE GUIDELINES

- Ensure commit messages are concise and clearly describe the changes made.
- Include relevant references to plans (e.g., `docs/troubleshooting/RCA_Plan.md`, `memory_blowup_RCA_steps.md`).
- Follow a consistent format (e.g., `feat(utils): add MemoryMonitor telemetry to StreamingDistanceMatrix`).

## WORKFLOW & RELEASE RULES
- General Workflow
    - **ALWAYS** activate the virtual environment (`source .venv/bin/activate`) before running any Python scripts or commands.
    - **YOU** (the AI) must run all commands, monitor terminal outputs, and handle errors immediately.
    - **I** (the user) do not need to be copying or pasting anything at any point, unless explicitly requested.
    - **YOU** are forbidden from presenting code in any way other than executable code blocks with Run/Cancel buttons.
    - **ALWAYS** perform static analysis (Codacy) after modifying any code.
    - When addressing memory blowup issues (reference `docs/troubleshooting/memory_blowup_RCA_steps.md`), prioritize steady but significant improvements, maintaining control for easy reversion and repair. Ensure alignment with Taskmaster planning and Agent Instruction files.
    - **Workflow for Executing Plans**: Locate relevant plans (e.g., RCA-related docs and strategy plans), extract actionable steps, and execute them. Begin with builds/tests and documentation updates. Open troubleshooting plans to

## 🚨 CI WORKFLOW NOTES
-