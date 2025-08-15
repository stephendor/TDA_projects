#include "tda/algorithms/streaming_cech.hpp"
#include "tda/algorithms/streaming_vr.hpp"
#include "tda/utils/streaming_distance_matrix.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <iomanip>
using std::strcmp;

using tda::algorithms::StreamingCechComplex;
using tda::algorithms::StreamingVRComplex;

static StreamingCechComplex::PointContainer make_points(size_t n, size_t d, uint32_t seed=42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    StreamingCechComplex::PointContainer pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> p(d);
        for (size_t j = 0; j < d; ++j) p[j] = dist(rng);
        pts.emplace_back(std::move(p));
    }
    return pts;
}

static std::string fnv1a64(const std::string &s) {
    const uint64_t FNV_OFFSET = 1469598103934665603ull;
    const uint64_t FNV_PRIME  = 1099511628211ull;
    uint64_t h = FNV_OFFSET;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= FNV_PRIME;
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << h;
    return oss.str();
}

int main(int argc, char** argv) {
    // Defaults
    size_t n = 2000, d = 3; // keep CI fast by default
    double radius = 0.9;     // Čech radius
    double epsilon = 0.9;    // VR epsilon
    int maxDim = 2;
    int maxNeighbors = 64;
    int block = 64;
    uint32_t seed = 42;
    const char* csvPath = nullptr;
    const char* jsonPath = nullptr; // JSONL
    const char* adjHistCsv = nullptr;           // export adjacency histogram (current run)
    const char* adjHistCsvBaseline = nullptr;   // export baseline adjacency histogram (baseline run)
    const char* baselineJsonOut = nullptr;      // optional: where the separate-process baseline writes its JSONL
    const char* pointsCsvPath = nullptr;        // optional: read input points from CSV file (overrides n/d)
    bool dmOnly = false;
    std::string modeSel = "cech"; // cech|vr
    int baseline_compare = 0; // if 1 and not dmOnly, run a baseline (no soft cap, serial threshold)
    int baseline_separate_process = 0; // if 1, run baseline in a separate process to avoid cumulative memory
    int baseline_maxDim = -1; // if >=0, override maxDim for baseline run
    // SDM soft cap for threshold mode
    int soft_knn_cap = 0; // 0 disables
    int softcap_local_merge = 0; // EXPERIMENTAL: keep default off; enable with 1 to try bounded local-merge
    // Early-stop / parallel controls
    size_t max_blocks = 0;
    size_t max_pairs = 0;
    double time_limit = 0.0;
    bool par_thresh = true;

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) { n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10)); }
        else if (!strcmp(argv[i], "--d") && i+1 < argc) { d = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10)); }
        else if (!strcmp(argv[i], "--radius") && i+1 < argc) { radius = std::strtod(argv[++i], nullptr); }
        else if (!strcmp(argv[i], "--epsilon") && i+1 < argc) { epsilon = std::strtod(argv[++i], nullptr); }
        else if (!strcmp(argv[i], "--maxDim") && i+1 < argc) { maxDim = std::atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--maxNeighbors") && i+1 < argc) { maxNeighbors = std::atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--block") && i+1 < argc) { block = std::atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) { seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10)); }
    else if (!strcmp(argv[i], "--csv") && i+1 < argc) { csvPath = argv[++i]; }
    else if (!strcmp(argv[i], "--json") && i+1 < argc) { jsonPath = argv[++i]; }
    else if (!strcmp(argv[i], "--adj-hist-csv") && i+1 < argc) { adjHistCsv = argv[++i]; }
    else if (!strcmp(argv[i], "--adj-hist-csv-baseline") && i+1 < argc) { adjHistCsvBaseline = argv[++i]; }
    else if (!strcmp(argv[i], "--baseline-json-out") && i+1 < argc) { baselineJsonOut = argv[++i]; }
    else if (!strcmp(argv[i], "--points-csv") && i+1 < argc) { pointsCsvPath = argv[++i]; }
    else if (!strcmp(argv[i], "--dm-only")) { dmOnly = true; }
    else if (!strcmp(argv[i], "--mode") && i+1 < argc) { modeSel = argv[++i]; }
    else if (!strcmp(argv[i], "--cuda") && i+1 < argc) { int cu = std::atoi(argv[++i]); if (cu) setenv("TDA_USE_CUDA","1",1); }
    else if (!strcmp(argv[i], "--max-blocks") && i+1 < argc) { max_blocks = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10)); }
    else if (!strcmp(argv[i], "--max-pairs") && i+1 < argc) { max_pairs = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10)); }
    else if (!strcmp(argv[i], "--time-limit") && i+1 < argc) { time_limit = std::strtod(argv[++i], nullptr); }
    else if (!strcmp(argv[i], "--parallel-threshold") && i+1 < argc) { par_thresh = std::atoi(argv[++i]) != 0; }
    else if (!strcmp(argv[i], "--soft-knn-cap") && i+1 < argc) { soft_knn_cap = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--softcap-local-merge") && i+1 < argc) { softcap_local_merge = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--baseline-compare") && i+1 < argc) { baseline_compare = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--baseline-separate-process") && i+1 < argc) { baseline_separate_process = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--baseline-maxDim") && i+1 < argc) { baseline_maxDim = std::atoi(argv[++i]); }
        else {
            // Positional fallback for backwards compat: n d
            if (i == 1) { n = static_cast<size_t>(std::strtoull(argv[i], nullptr, 10)); }
            else if (i == 2) { d = static_cast<size_t>(std::strtoull(argv[i], nullptr, 10)); }
        }
    }

    auto pts = StreamingCechComplex::PointContainer{};
    if (pointsCsvPath && std::strlen(pointsCsvPath) > 0) {
        // Read points from CSV; each line: x1,x2,...,xd
        std::ifstream pin(pointsCsvPath);
        if (!pin.good()) {
            std::cerr << "Failed to open points CSV: " << pointsCsvPath << "\n";
            return 2;
        }
        std::string line;
        size_t inferred_d = 0;
        while (std::getline(pin, line)) {
            if (line.empty()) continue;
            std::vector<double> coords;
            coords.reserve(8);
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) {
                    try {
                        coords.push_back(std::stod(tok));
                    } catch (...) {
                        // treat non-numeric as 0
                        coords.push_back(0.0);
                    }
                } else {
                    coords.push_back(0.0);
                }
            }
            if (coords.empty()) continue;
            if (inferred_d == 0) inferred_d = coords.size();
            // normalize dimension: pad or trim to inferred_d
            if (coords.size() < inferred_d) coords.resize(inferred_d, 0.0);
            else if (coords.size() > inferred_d) coords.resize(inferred_d);
            pts.emplace_back(std::move(coords));
        }
        if (pts.empty()) {
            std::cerr << "No points parsed from CSV: " << pointsCsvPath << "\n";
            return 3;
        }
        // Override n and d from parsed data
        n = pts.size();
        d = pts.front().size();
        std::cout << "[input] Loaded " << n << " points of dimension " << d << " from CSV: " << pointsCsvPath << "\n";
    } else {
        pts = make_points(n, d, seed);
    }

    // Result vars
    size_t dm_blocks = 0, dm_edges = 0;
    double dm_peak_mb = 0.0;
    double build_secs = 0.0;
    double build_peak_mb = 0.0;
    size_t simplices = 0;
    // SimplexPool telemetry (aggregate)
    size_t pool_total_blocks = 0;
    size_t pool_free_blocks = 0;
    double pool_fragmentation = 0.0;
    size_t pool_pages = 0;
    size_t pool_blocks_per_page = 0;
    std::string pool_bucket_stats_compact; // e.g., "1:10/2;2:5/1"
    // DM early-stop diagnostics (defaults; populated in DM-only path)
    size_t dm_blocks_stats = 0;
    size_t dm_pairs_stats = 0;
    int dm_stopped_time = 0, dm_stopped_pairs = 0, dm_stopped_blocks = 0;
    size_t dm_last_i0 = 0, dm_last_j0 = 0;
    // Manifest & soft-cap telemetry
    std::string manifest_hash;
    size_t attempted_count = 0;
    std::string attempted_kind;
    // Soft-cap telemetry
    size_t overshoot_sum = 0;
    size_t overshoot_max = 0;
    // Normalize mode selector
    for (auto& c : modeSel) c = static_cast<char>(::tolower(c));
    bool useVR = (modeSel == "vr");
    std::string mode = dmOnly ? "DMOnly" : (useVR ? "VR" : "Cech");

    if (dmOnly) {
        // Stream only the distance matrix edges/blocks under threshold 2*radius
    tda::utils::StreamingDMConfig cfg;
        cfg.block_size = static_cast<size_t>(block);
        cfg.symmetric = true;
        cfg.max_distance = 2.0 * radius;
    cfg.use_parallel = true;
    cfg.max_blocks = max_blocks;
    cfg.max_pairs = max_pairs;
    cfg.time_limit_seconds = time_limit;
    cfg.enable_parallel_threshold = par_thresh;
    cfg.edge_callback_threadsafe = true; // counting only, safe
    cfg.knn_cap_per_vertex_soft = static_cast<size_t>(soft_knn_cap > 0 ? soft_knn_cap : 0);
    cfg.softcap_local_merge = (softcap_local_merge != 0);
    // optional CUDA
    if (const char* CUDA_ENV = std::getenv("TDA_USE_CUDA"); CUDA_ENV && std::strlen(CUDA_ENV) > 0 && CUDA_ENV[0] == '1') cfg.use_cuda = true;

        tda::utils::StreamingDistanceMatrix dm(cfg);
        dm.onEdge([&](size_t, size_t, double){ /* no-op, just count in stats */ });
        auto t0 = std::chrono::steady_clock::now();
        auto stats = dm.process(pts);
        auto t1 = std::chrono::steady_clock::now();
        dm_blocks = stats.total_blocks;
        dm_edges = stats.emitted_edges;
        dm_peak_mb = stats.peak_memory_bytes / (1024.0 * 1024.0);
        overshoot_sum = stats.softcap_overshoot_sum;
        overshoot_max = stats.softcap_overshoot_max;
        build_secs = std::chrono::duration<double>(t1 - t0).count();
        build_peak_mb = 0.0;
        simplices = 0;
        // capture for JSONL later
        dm_blocks_stats = stats.total_blocks;
        dm_pairs_stats = stats.total_pairs;
        dm_stopped_time = stats.stopped_by_time ? 1 : 0;
        dm_stopped_pairs = stats.stopped_by_pairs ? 1 : 0;
        dm_stopped_blocks = stats.stopped_by_blocks ? 1 : 0;
        dm_last_i0 = stats.last_block_i0;
        dm_last_j0 = stats.last_block_j0;
        if (soft_knn_cap > 0) {
            std::cout << "  [softcap] overshoot_sum=" << stats.softcap_overshoot_sum
                      << ", overshoot_max=" << stats.softcap_overshoot_max << std::endl;
        }
    } else {
        // Full streaming complex path: Čech or VR based on modeSel
        if (!useVR) {
            // Čech
            StreamingCechComplex::Config cfg;
            cfg.radius = radius;
            cfg.maxDimension = static_cast<size_t>(maxDim);
            cfg.maxNeighbors = static_cast<size_t>(maxNeighbors);
            cfg.useAdaptiveRadius = false;
            cfg.dm.block_size = static_cast<size_t>(block);
            cfg.dm.symmetric = true;
            cfg.dm.max_distance = -1.0; // set internally from radius
            // Propagate early-stop and parallel controls to the DM used inside Cech
            cfg.dm.max_blocks = max_blocks;
            cfg.dm.max_pairs = max_pairs;
            cfg.dm.time_limit_seconds = time_limit;
            cfg.dm.enable_parallel_threshold = par_thresh;
            cfg.dm.edge_callback_threadsafe = true; // guarded updates in builder
            cfg.dm.knn_cap_per_vertex_soft = static_cast<size_t>(soft_knn_cap > 0 ? soft_knn_cap : 0);
            cfg.dm.softcap_local_merge = (softcap_local_merge != 0);
            if (const char* CUDA_ENV2 = std::getenv("TDA_USE_CUDA"); CUDA_ENV2 && std::strlen(CUDA_ENV2) > 0 && CUDA_ENV2[0] == '1') cfg.dm.use_cuda = true;

            // Run primary build in a limited scope to free memory before optional baseline compare
            StreamingCechComplex::BuildStats b;
            tda::utils::StreamingDMStats dm;
            {
                StreamingCechComplex cech(cfg);
                auto r1 = cech.initialize(pts);
                if (r1.has_error()) { std::cerr << r1.error() << "\n"; return 1; }
                auto r2 = cech.computeComplex();
                if (r2.has_error()) { std::cerr << r2.error() << "\n"; return 1; }
                dm = cech.getStreamingStats();
                b  = cech.getBuildStats();
                // Optional: export adjacency histogram for QA (do it before cech is destroyed)
                if (adjHistCsv) {
                    std::ofstream out(adjHistCsv);
                    if (!out.good()) {
                        std::cerr << "Failed to open adj histogram file: " << adjHistCsv << "\n";
                    } else {
                        out << "degree,count\n";
                        for (size_t deg = 0; deg < b.adjacency_histogram.size(); ++deg) {
                            out << deg << "," << b.adjacency_histogram[deg] << "\n";
                        }
                    }
                }
                // Capture pool stats (aggregate + compact per-bucket string)
                pool_total_blocks = b.pool_total_blocks;
                pool_free_blocks = b.pool_free_blocks;
                pool_fragmentation = b.pool_fragmentation;
                pool_pages = b.pool_pages;
                pool_blocks_per_page = b.pool_blocks_per_page;
                if (!b.pool_bucket_stats.empty()) {
                    std::ostringstream oss;
                    // Order by arity ascending for stability, limit to first 12 entries to keep it short
                    size_t count = 0;
                    for (const auto& ent : b.pool_bucket_stats) {
                        if (count++) oss << ";";
                        oss << ent.first << ":" << ent.second.first << "/" << ent.second.second;
                        if (count >= 12) break;
                    }
                    pool_bucket_stats_compact = oss.str();
                } else {
                    pool_bucket_stats_compact.clear();
                }
            } // cech destroyed here to reduce overlapping memory with baseline
            dm_blocks = dm.total_blocks;
            dm_edges = dm.emitted_edges;
            dm_peak_mb = dm.peak_memory_bytes / (1024.0 * 1024.0);
            overshoot_sum = dm.softcap_overshoot_sum;
            overshoot_max = dm.softcap_overshoot_max;
            build_secs = b.elapsed_seconds;
            build_peak_mb = b.peak_memory_bytes / (1024.0 * 1024.0);
            simplices = b.num_simplices_built;
            attempted_count = simplices;
            attempted_kind = "simplices";
            if (soft_knn_cap > 0) {
                std::cout << "  [softcap] overshoot_sum=" << dm.softcap_overshoot_sum
                          << ", overshoot_max=" << dm.softcap_overshoot_max << std::endl;
            }
            // Optional: run a baseline (no soft cap, serial threshold) for accuracy QA and compare
            if (baseline_compare && soft_knn_cap > 0) {
                StreamingCechComplex::Config baseCfg = cfg;
                baseCfg.dm.knn_cap_per_vertex_soft = 0; // disable soft cap
                baseCfg.dm.enable_parallel_threshold = false; // race-free deterministic
                baseCfg.dm.softcap_local_merge = false;
                if (baseline_maxDim >= 0) baseCfg.maxDimension = static_cast<size_t>(baseline_maxDim);
                if (baseline_separate_process) {
                    // Run separate process to avoid cumulative memory
                    char tmpPath[256];
                    if (baselineJsonOut && std::strlen(baselineJsonOut) > 0) {
                        std::snprintf(tmpPath, sizeof(tmpPath), "%s", baselineJsonOut);
                    } else {
                        std::snprintf(tmpPath, sizeof(tmpPath), "/tmp/tda_baseline_%ld_%d.jsonl", static_cast<long>(std::time(nullptr)), static_cast<int>(::getpid()));
                    }
                    std::ostringstream cmd;
                    cmd << argv[0]
                        << " --n " << n
                        << " --d " << d
                        << " --radius " << radius
                        << " --maxDim " << baseCfg.maxDimension
                        << " --maxNeighbors " << maxNeighbors
                        << " --block " << block
                        << " --seed " << seed
                        << " --parallel-threshold 0"
                        << " --soft-knn-cap 0"
                        << " --json " << tmpPath;
                    if (adjHistCsvBaseline) {
                        cmd << " --adj-hist-csv " << adjHistCsvBaseline;
                    }
                    int rc = std::system(cmd.str().c_str());
                    if (rc != 0) {
                        std::cerr << "Baseline subprocess failed with code " << rc << "\n";
                    } else {
                        std::ifstream jin(tmpPath);
                        std::string line, last;
                        while (std::getline(jin, line)) { if (!line.empty()) last = line; }
                        size_t dm_pos = last.find("\"dm_edges\":");
                        size_t sx_pos = last.find("\"simplices\":");
                        size_t dm_val = 0; size_t sx_val = 0;
                        if (dm_pos != std::string::npos) dm_val = std::strtoull(last.c_str() + dm_pos + 11, nullptr, 10);
                        if (sx_pos != std::string::npos) sx_val = std::strtoull(last.c_str() + sx_pos + 12, nullptr, 10);
                        long delta_edges = static_cast<long>(dm_edges) - static_cast<long>(dm_val);
                        long delta_sx   = static_cast<long>(simplices) - static_cast<long>(sx_val);
                        std::cout << "Baseline (separate, no soft cap, serial) dm_edges=" << dm_val
                                  << ", simplices=" << sx_val
                                  << ", delta_edges=" << delta_edges
                                  << ", delta_simplices=" << delta_sx << std::endl;
                    }
                } else {
                    StreamingCechComplex::BuildStats b2;
                    tda::utils::StreamingDMStats dm2;
                    {
                        StreamingCechComplex cechBase(baseCfg);
                        auto rb1 = cechBase.initialize(pts);
                        if (rb1.has_error()) { std::cerr << rb1.error() << "\n"; return 1; }
                        auto rb2 = cechBase.computeComplex();
                        if (rb2.has_error()) { std::cerr << rb2.error() << "\n"; return 1; }
                        dm2 = cechBase.getStreamingStats();
                        b2  = cechBase.getBuildStats();
                    }
                    long delta_edges = static_cast<long>(dm_edges) - static_cast<long>(dm2.emitted_edges);
                    long delta_sx   = static_cast<long>(simplices) - static_cast<long>(b2.num_simplices_built);
                    std::cout << "Baseline (no soft cap, serial) dm_edges=" << dm2.emitted_edges
                              << ", simplices=" << b2.num_simplices_built
                              << ", delta_edges=" << delta_edges
                              << ", delta_simplices=" << delta_sx << std::endl;
                    if (adjHistCsvBaseline) {
                        std::ofstream out2(adjHistCsvBaseline);
                        if (!out2.good()) {
                            std::cerr << "Failed to open baseline adj histogram file: " << adjHistCsvBaseline << "\n";
                        } else {
                            out2 << "degree,count\n";
                            for (size_t deg = 0; deg < b2.adjacency_histogram.size(); ++deg) {
                                out2 << deg << "," << b2.adjacency_histogram[deg] << "\n";
                            }
                        }
                    }
                }
            }
        } else {
            // VR
            StreamingVRComplex::Config cfg;
            cfg.epsilon = epsilon;
            cfg.maxDimension = static_cast<size_t>(maxDim);
            cfg.maxNeighbors = static_cast<size_t>(maxNeighbors);
            cfg.dm.block_size = static_cast<size_t>(block);
            cfg.dm.symmetric = true;
            cfg.dm.max_distance = -1.0; // will be set from epsilon internally
            cfg.dm.max_blocks = max_blocks;
            cfg.dm.max_pairs = max_pairs;
            cfg.dm.time_limit_seconds = time_limit;
            cfg.dm.enable_parallel_threshold = par_thresh;
            cfg.dm.edge_callback_threadsafe = true;
            cfg.dm.knn_cap_per_vertex_soft = static_cast<size_t>(soft_knn_cap > 0 ? soft_knn_cap : 0);
            cfg.dm.softcap_local_merge = (softcap_local_merge != 0);
            if (const char* CUDA_ENV3 = std::getenv("TDA_USE_CUDA"); CUDA_ENV3 && std::strlen(CUDA_ENV3) > 0 && CUDA_ENV3[0] == '1') cfg.dm.use_cuda = true;

            StreamingVRComplex::BuildStats b;
            tda::utils::StreamingDMStats dm;
            {
                StreamingVRComplex vr(cfg);
                auto r1 = vr.initialize(pts);
                if (r1.has_error()) { std::cerr << r1.error() << "\n"; return 1; }
                auto r2 = vr.computeComplex();
                if (r2.has_error()) { std::cerr << r2.error() << "\n"; return 1; }
                dm = vr.getStreamingStats();
                b  = vr.getBuildStats();
                if (adjHistCsv) {
                    std::ofstream out(adjHistCsv);
                    if (!out.good()) {
                        std::cerr << "Failed to open adj histogram file: " << adjHistCsv << "\n";
                    } else {
                        out << "degree,count\n";
                        for (size_t deg = 0; deg < b.adjacency_histogram.size(); ++deg) {
                            out << deg << "," << b.adjacency_histogram[deg] << "\n";
                        }
                    }
                }
                // Capture pool stats (aggregate + compact per-bucket string)
                pool_total_blocks = b.pool_total_blocks;
                pool_free_blocks = b.pool_free_blocks;
                pool_fragmentation = b.pool_fragmentation;
                pool_pages = b.pool_pages;
                pool_blocks_per_page = b.pool_blocks_per_page;
                if (!b.pool_bucket_stats.empty()) {
                    std::ostringstream oss;
                    size_t count = 0;
                    for (const auto& ent : b.pool_bucket_stats) {
                        if (count++) oss << ";";
                        oss << ent.first << ":" << ent.second.first << "/" << ent.second.second;
                        if (count >= 12) break;
                    }
                    pool_bucket_stats_compact = oss.str();
                } else {
                    pool_bucket_stats_compact.clear();
                }
            }
            dm_blocks = dm.total_blocks;
            dm_edges = dm.emitted_edges;
            dm_peak_mb = dm.peak_memory_bytes / (1024.0 * 1024.0);
            overshoot_sum = dm.softcap_overshoot_sum;
            overshoot_max = dm.softcap_overshoot_max;
            build_secs = b.elapsed_seconds;
            build_peak_mb = b.peak_memory_bytes / (1024.0 * 1024.0);
            simplices = b.num_simplices_built;
            attempted_count = simplices;
            attempted_kind = "simplices";
            if (soft_knn_cap > 0) {
                std::cout << "  [softcap] overshoot_sum=" << dm.softcap_overshoot_sum
                          << ", overshoot_max=" << dm.softcap_overshoot_max << std::endl;
            }
            if (baseline_compare && soft_knn_cap > 0) {
                StreamingVRComplex::Config baseCfg = cfg;
                baseCfg.dm.knn_cap_per_vertex_soft = 0;
                baseCfg.dm.enable_parallel_threshold = false;
                baseCfg.dm.softcap_local_merge = false;
                if (baseline_maxDim >= 0) baseCfg.maxDimension = static_cast<size_t>(baseline_maxDim);
                if (baseline_separate_process) {
                    char tmpPath[256];
                    if (baselineJsonOut && std::strlen(baselineJsonOut) > 0) {
                        std::snprintf(tmpPath, sizeof(tmpPath), "%s", baselineJsonOut);
                    } else {
                        std::snprintf(tmpPath, sizeof(tmpPath), "/tmp/tda_baseline_%ld_%d.jsonl", static_cast<long>(std::time(nullptr)), static_cast<int>(::getpid()));
                    }
                    std::ostringstream cmd;
                    cmd << argv[0]
                        << " --mode vr"
                        << " --n " << n
                        << " --d " << d
                        << " --epsilon " << epsilon
                        << " --maxDim " << baseCfg.maxDimension
                        << " --maxNeighbors " << maxNeighbors
                        << " --block " << block
                        << " --seed " << seed
                        << " --parallel-threshold 0"
                        << " --soft-knn-cap 0"
                        << " --json " << tmpPath;
                    if (adjHistCsvBaseline) {
                        cmd << " --adj-hist-csv " << adjHistCsvBaseline;
                    }
                    int rc = std::system(cmd.str().c_str());
                    if (rc != 0) {
                        std::cerr << "Baseline subprocess failed with code " << rc << "\n";
                    } else {
                        std::ifstream jin(tmpPath);
                        std::string line, last;
                        while (std::getline(jin, line)) { if (!line.empty()) last = line; }
                        size_t dm_pos = last.find("\"dm_edges\":");
                        size_t sx_pos = last.find("\"simplices\":");
                        size_t dm_val = 0; size_t sx_val = 0;
                        if (dm_pos != std::string::npos) dm_val = std::strtoull(last.c_str() + dm_pos + 11, nullptr, 10);
                        if (sx_pos != std::string::npos) sx_val = std::strtoull(last.c_str() + sx_pos + 12, nullptr, 10);
                        long delta_edges = static_cast<long>(dm_edges) - static_cast<long>(dm_val);
                        long delta_sx   = static_cast<long>(simplices) - static_cast<long>(sx_val);
                        std::cout << "Baseline (VR separate, no soft cap, serial) dm_edges=" << dm_val
                                  << ", simplices=" << sx_val
                                  << ", delta_edges=" << delta_edges
                                  << ", delta_simplices=" << delta_sx << std::endl;
                    }
                } else {
                    StreamingVRComplex::BuildStats b2;
                    tda::utils::StreamingDMStats dm2;
                    {
                        StreamingVRComplex vrBase(baseCfg);
                        auto rb1 = vrBase.initialize(pts);
                        if (rb1.has_error()) { std::cerr << rb1.error() << "\n"; return 1; }
                        auto rb2 = vrBase.computeComplex();
                        if (rb2.has_error()) { std::cerr << rb2.error() << "\n"; return 1; }
                        dm2 = vrBase.getStreamingStats();
                        b2  = vrBase.getBuildStats();
                    }
                    long delta_edges = static_cast<long>(dm_edges) - static_cast<long>(dm2.emitted_edges);
                    long delta_sx   = static_cast<long>(simplices) - static_cast<long>(b2.num_simplices_built);
                    std::cout << "Baseline (VR no soft cap, serial) dm_edges=" << dm2.emitted_edges
                              << ", simplices=" << b2.num_simplices_built
                              << ", delta_edges=" << delta_edges
                              << ", delta_simplices=" << delta_sx << std::endl;
                    if (adjHistCsvBaseline) {
                        std::ofstream out2(adjHistCsvBaseline);
                        if (!out2.good()) {
                            std::cerr << "Failed to open baseline adj histogram file: " << adjHistCsvBaseline << "\n";
                        } else {
                            out2 << "degree,count\n";
                            for (size_t deg = 0; deg < b2.adjacency_histogram.size(); ++deg) {
                                out2 << deg << "," << b2.adjacency_histogram[deg] << "\n";
                            }
                        }
                    }
                }
            }
        }
    }

    // Console output
    std::cout << "StreamingPerf mode=" << mode
              << ", n=" << n << ", d=" << d
              << (useVR ? ", epsilon=" : ", radius=") << (useVR ? epsilon : radius)
              << ", maxDim=" << maxDim
              << ", maxNeighbors=" << maxNeighbors
              << ", block=" << block
              << ", softKnnCap=" << soft_knn_cap
              << ", softcapLocalMerge=" << softcap_local_merge
              << ", baselineCompare=" << baseline_compare
              << ", maxBlocks=" << max_blocks
              << ", maxPairs=" << max_pairs
              << ", timeLimit=" << time_limit
              << ", DM blocks=" << dm_blocks
              << ", DM edges=" << dm_edges
              << ", DM peakMB=" << dm_peak_mb
              << ", Build secs=" << build_secs
              << ", Build peakMB=" << build_peak_mb
              << ", Simplices=" << simplices
              << std::endl;

    // CSV logging (append)
    if (csvPath) {
        bool writeHeader = false;
        {
            std::ifstream fin(csvPath, std::ios::in);
            writeHeader = !fin.good();
        }
        std::ofstream fout(csvPath, std::ios::app);
        if (writeHeader) {
            fout << "timestamp,n,d,mode,threshold,maxDim,maxNeighbors,block,maxBlocks,maxPairs,timeLimit,dm_blocks,dm_edges,dm_peak_mb,build_secs,build_peak_mb,simplices,overshoot_sum,overshoot_max,pool_total_blocks,pool_free_blocks,pool_fragmentation\n";
        }
        // ISO-ish timestamp
        std::time_t t = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
        double thr = useVR ? epsilon : radius;
    fout << buf << "," << n << "," << d << "," << mode << "," << thr << ","
             << maxDim << "," << maxNeighbors << "," << block << "," << max_blocks << "," << max_pairs << "," << time_limit << "," << dm_blocks << ","
         << dm_edges << "," << dm_peak_mb << "," << build_secs << "," << build_peak_mb << ","
         << simplices << "," << overshoot_sum << "," << overshoot_max << ","
         << pool_total_blocks << "," << pool_free_blocks << "," << pool_fragmentation << "\n";
    }

    // JSONL logging (append one JSON object per line)
    if (jsonPath) {
        std::ofstream jout(jsonPath, std::ios::app);
        std::time_t t = std::time(nullptr);
        char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    jout << "{\"timestamp\":\"" << buf << "\",";
    jout << "\"n\":" << n << ",\"d\":" << d << ",\"mode\":\"" << mode << "\",";
       // Config echo
    if (useVR) { jout << "\"epsilon\":" << epsilon << ","; } else { jout << "\"radius\":" << radius << ","; }
    jout << "\"maxDim\":" << maxDim << ",\"maxNeighbors\":" << maxNeighbors
           << ",\"block\":" << block << ",\"maxBlocks\":" << max_blocks << ",\"maxPairs\":" << max_pairs
           << ",\"timeLimit\":" << time_limit << ",\"parallel_threshold\":" << (par_thresh ? 1 : 0) << ",";
       // Core telemetry (underscore keys maintained for backward compatibility)
       jout << "\"dm_blocks\":" << dm_blocks << ",\"dm_edges\":" << dm_edges
           << ",\"dm_peak_mb\":" << dm_peak_mb << ",\"build_secs\":" << build_secs
           << ",\"build_peak_mb\":" << build_peak_mb << ",\"simplices\":" << simplices
           << ",\"overshoot_sum\":" << overshoot_sum << ",\"overshoot_max\":" << overshoot_max << ",";
       // Duplicate fields with camelCase expected by analyzer
    jout << "\"dm_peakMB\":" << dm_peak_mb << ",\"rss_peakMB\":" << build_peak_mb
           << ",\"softcap_overshoot_sum\":" << overshoot_sum
           << ",\"softcap_overshoot_max\":" << overshoot_max << ","
           // Pool telemetry (aggregate)
           << "\"pool_total_blocks\":" << pool_total_blocks << ",\"pool_free_blocks\":" << pool_free_blocks
           << ",\"pool_fragmentation\":" << pool_fragmentation
           // Optional page info (if available in BuildStats)
           << ",\"pool_pages\":" << pool_pages
           << ",\"pool_blocks_per_page\":" << pool_blocks_per_page
           // Early-stop diagnostics (DM-only path populates these)
           << ",\"dm_total_blocks\":" << (dmOnly ? dm_blocks_stats : 0)
           << ",\"dm_total_pairs\":" << (dmOnly ? dm_pairs_stats : 0)
           << ",\"dm_stopped_by_time\":" << (dmOnly ? dm_stopped_time : 0)
           << ",\"dm_stopped_by_pairs\":" << (dmOnly ? dm_stopped_pairs : 0)
           << ",\"dm_stopped_by_blocks\":" << (dmOnly ? dm_stopped_blocks : 0)
           << ",\"dm_last_block_i0\":" << (dmOnly ? dm_last_i0 : 0)
           << ",\"dm_last_block_j0\":" << (dmOnly ? dm_last_j0 : 0)
           // Manifest hash & attempted recompute count
           << ",\"attempted_count\":" << attempted_count
           << ",\"attempted_kind\":\"" << attempted_kind << "\""
           << ",\"manifest_hash\":\"";
        {
            std::ostringstream man;
            man << "mode=" << mode
                << ",n=" << n << ",d=" << d
                << (useVR ? ",epsilon=" : ",radius=") << (useVR ? epsilon : radius)
                << ",maxDim=" << maxDim << ",maxNeighbors=" << maxNeighbors
                << ",block=" << block
                << ",soft_knn_cap=" << soft_knn_cap
                << ",parallel_threshold=" << (par_thresh ? 1 : 0)
                << ",softcap_local_merge=" << softcap_local_merge
                << ",maxBlocks=" << max_blocks << ",maxPairs=" << max_pairs
                << ",timeLimit=" << time_limit
                << ",seed=" << seed
                << ",use_cuda=" << (std::getenv("TDA_USE_CUDA") ? 1 : 0);
            manifest_hash = fnv1a64(man.str());
        }
        jout << manifest_hash << "\""
           // Optional compact per-bucket stats as string: "arity:total/free;..." (empty for now)
           << ",\"pool_bucket_stats_compact\":\"" << pool_bucket_stats_compact << "\""
           << "}" << "\n";
    }

    return 0;
}
