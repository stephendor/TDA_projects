#include "tda/algorithms/streaming_cech.hpp"
#include "tda/utils/streaming_distance_matrix.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
using std::strcmp;

using tda::algorithms::StreamingCechComplex;

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

int main(int argc, char** argv) {
    // Defaults
    size_t n = 2000, d = 3; // keep CI fast by default
    double radius = 0.9;
    int maxDim = 2;
    int maxNeighbors = 64;
    int block = 64;
    uint32_t seed = 42;
    const char* csvPath = nullptr;
    const char* jsonPath = nullptr; // JSONL
    const char* adjHistCsv = nullptr;           // export adjacency histogram (current run)
    const char* adjHistCsvBaseline = nullptr;   // export baseline adjacency histogram (baseline run)
    const char* baselineJsonOut = nullptr;      // optional: where the separate-process baseline writes its JSONL
    bool dmOnly = false;
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
        else if (!strcmp(argv[i], "--maxDim") && i+1 < argc) { maxDim = std::atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--maxNeighbors") && i+1 < argc) { maxNeighbors = std::atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--block") && i+1 < argc) { block = std::atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) { seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10)); }
    else if (!strcmp(argv[i], "--csv") && i+1 < argc) { csvPath = argv[++i]; }
    else if (!strcmp(argv[i], "--json") && i+1 < argc) { jsonPath = argv[++i]; }
    else if (!strcmp(argv[i], "--adj-hist-csv") && i+1 < argc) { adjHistCsv = argv[++i]; }
    else if (!strcmp(argv[i], "--adj-hist-csv-baseline") && i+1 < argc) { adjHistCsvBaseline = argv[++i]; }
    else if (!strcmp(argv[i], "--baseline-json-out") && i+1 < argc) { baselineJsonOut = argv[++i]; }
    else if (!strcmp(argv[i], "--dm-only")) { dmOnly = true; }
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

    auto pts = make_points(n, d, seed);

    // Result vars
    size_t dm_blocks = 0, dm_edges = 0;
    double dm_peak_mb = 0.0;
    double build_secs = 0.0;
    double build_peak_mb = 0.0;
    size_t simplices = 0;
    // Soft-cap telemetry
    size_t overshoot_sum = 0;
    size_t overshoot_max = 0;
    std::string mode = dmOnly ? "DMOnly" : "Cech";

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

        tda::utils::StreamingDistanceMatrix dm(cfg);
        dm.onEdge([&](size_t, size_t, double){ /* no-op, just count in stats */ });
    auto stats = dm.process(pts);
        dm_blocks = stats.total_blocks;
        dm_edges = stats.emitted_edges;
        dm_peak_mb = stats.peak_memory_bytes / (1024.0 * 1024.0);
    overshoot_sum = stats.softcap_overshoot_sum;
    overshoot_max = stats.softcap_overshoot_max;
        build_secs = 0.0;
        build_peak_mb = 0.0;
        simplices = 0;
        if (soft_knn_cap > 0) {
            std::cout << "  [softcap] overshoot_sum=" << stats.softcap_overshoot_sum
                      << ", overshoot_max=" << stats.softcap_overshoot_max << std::endl;
        }
    } else {
        // Full streaming Čech path
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
        } // cech destroyed here to reduce overlapping memory with baseline
    dm_blocks = dm.total_blocks;
    dm_edges = dm.emitted_edges;
    dm_peak_mb = dm.peak_memory_bytes / (1024.0 * 1024.0);
    overshoot_sum = dm.softcap_overshoot_sum;
    overshoot_max = dm.softcap_overshoot_max;
        build_secs = b.elapsed_seconds;
        build_peak_mb = b.peak_memory_bytes / (1024.0 * 1024.0);
        simplices = b.num_simplices_built;
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
                // Prepare temp JSONL file path
                char tmpPath[256];
                if (baselineJsonOut && std::strlen(baselineJsonOut) > 0) {
                    std::snprintf(tmpPath, sizeof(tmpPath), "%s", baselineJsonOut);
                } else {
                    std::snprintf(tmpPath, sizeof(tmpPath), "/tmp/tda_baseline_%ld_%d.jsonl", static_cast<long>(std::time(nullptr)), static_cast<int>(::getpid()));
                }
                std::ostringstream cmd;
                // Use argv[0] as the same binary
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
                    // Parse last line of JSONL for dm_edges and simplices
                    std::ifstream jin(tmpPath);
                    std::string line, last;
                    while (std::getline(jin, line)) { if (!line.empty()) last = line; }
                    size_t dm_pos = last.find("\"dm_edges\":");
                    size_t sx_pos = last.find("\"simplices\":");
                    size_t dm_val = 0; size_t sx_val = 0;
                    if (dm_pos != std::string::npos) {
                        dm_val = std::strtoull(last.c_str() + dm_pos + 11, nullptr, 10);
                    }
                    if (sx_pos != std::string::npos) {
                        sx_val = std::strtoull(last.c_str() + sx_pos + 12, nullptr, 10);
                    }
                    long delta_edges = static_cast<long>(dm_edges) - static_cast<long>(dm_val);
                    long delta_sx   = static_cast<long>(simplices) - static_cast<long>(sx_val);
                    std::cout << "Baseline (separate, no soft cap, serial) dm_edges=" << dm_val
                              << ", simplices=" << sx_val
                              << ", delta_edges=" << delta_edges
                              << ", delta_simplices=" << delta_sx << std::endl;
                }
            } else {
                // In-process baseline run (may increase peak RSS due to allocator behavior)
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
    }

    // Console output
    std::cout << "StreamingCechPerf mode=" << mode
              << ", n=" << n << ", d=" << d
              << ", radius=" << radius
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
            fout << "timestamp,n,d,mode,radius,maxDim,maxNeighbors,block,maxBlocks,maxPairs,timeLimit,dm_blocks,dm_edges,dm_peak_mb,build_secs,build_peak_mb,simplices,overshoot_sum,overshoot_max\n";
        }
        // ISO-ish timestamp
        std::time_t t = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
        fout << buf << "," << n << "," << d << "," << mode << "," << radius << ","
             << maxDim << "," << maxNeighbors << "," << block << "," << max_blocks << "," << max_pairs << "," << time_limit << "," << dm_blocks << ","
             << dm_edges << "," << dm_peak_mb << "," << build_secs << "," << build_peak_mb << ","
             << simplices << "," << overshoot_sum << "," << overshoot_max << "\n";
    }

    // JSONL logging (append one JSON object per line)
    if (jsonPath) {
        std::ofstream jout(jsonPath, std::ios::app);
        std::time_t t = std::time(nullptr);
        char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    jout << "{\"timestamp\":\"" << buf << "\",";
        jout << "\"n\":" << n << ",\"d\":" << d << ",\"mode\":\"" << mode << "\",";
    jout << "\"radius\":" << radius << ",\"maxDim\":" << maxDim << ",\"maxNeighbors\":" << maxNeighbors << ",\"block\":" << block << ",\"maxBlocks\":" << max_blocks << ",\"maxPairs\":" << max_pairs << ",\"timeLimit\":" << time_limit << ",";
    jout << "\"dm_blocks\":" << dm_blocks << ",\"dm_edges\":" << dm_edges << ",\"dm_peak_mb\":" << dm_peak_mb << ",";
    jout << "\"build_secs\":" << build_secs << ",\"build_peak_mb\":" << build_peak_mb << ",\"simplices\":" << simplices
         << ",\"overshoot_sum\":" << overshoot_sum << ",\"overshoot_max\":" << overshoot_max << "}\n";
    }

    return 0;
}
