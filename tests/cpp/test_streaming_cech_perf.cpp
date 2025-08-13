#include "tda/algorithms/streaming_cech.hpp"
#include "tda/utils/streaming_distance_matrix.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <cstring>
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
    bool dmOnly = false;
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
    else if (!strcmp(argv[i], "--dm-only")) { dmOnly = true; }
    else if (!strcmp(argv[i], "--max-blocks") && i+1 < argc) { max_blocks = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10)); }
    else if (!strcmp(argv[i], "--max-pairs") && i+1 < argc) { max_pairs = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10)); }
    else if (!strcmp(argv[i], "--time-limit") && i+1 < argc) { time_limit = std::strtod(argv[++i], nullptr); }
    else if (!strcmp(argv[i], "--parallel-threshold") && i+1 < argc) { par_thresh = std::atoi(argv[++i]) != 0; }
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

        tda::utils::StreamingDistanceMatrix dm(cfg);
        dm.onEdge([&](size_t, size_t, double){ /* no-op, just count in stats */ });
        auto stats = dm.process(pts);
        dm_blocks = stats.total_blocks;
        dm_edges = stats.emitted_edges;
        dm_peak_mb = stats.peak_memory_bytes / (1024.0 * 1024.0);
        build_secs = 0.0;
        build_peak_mb = 0.0;
        simplices = 0;
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

        StreamingCechComplex cech(cfg);
        auto r1 = cech.initialize(pts);
        if (r1.has_error()) { std::cerr << r1.error() << "\n"; return 1; }
        auto r2 = cech.computeComplex();
        if (r2.has_error()) { std::cerr << r2.error() << "\n"; return 1; }

        auto dm = cech.getStreamingStats();
        auto b  = cech.getBuildStats();
        dm_blocks = dm.total_blocks;
        dm_edges = dm.emitted_edges;
        dm_peak_mb = dm.peak_memory_bytes / (1024.0 * 1024.0);
        build_secs = b.elapsed_seconds;
        build_peak_mb = b.peak_memory_bytes / (1024.0 * 1024.0);
        simplices = b.num_simplices_built;
    }

    // Console output
    std::cout << "StreamingCechPerf mode=" << mode
              << ", n=" << n << ", d=" << d
              << ", radius=" << radius
              << ", maxDim=" << maxDim
              << ", maxNeighbors=" << maxNeighbors
              << ", block=" << block
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
            fout << "timestamp,n,d,mode,radius,maxDim,maxNeighbors,block,maxBlocks,maxPairs,timeLimit,dm_blocks,dm_edges,dm_peak_mb,build_secs,build_peak_mb,simplices\n";
        }
        // ISO-ish timestamp
        std::time_t t = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    fout << buf << "," << n << "," << d << "," << mode << "," << radius << ","
         << maxDim << "," << maxNeighbors << "," << block << "," << max_blocks << "," << max_pairs << "," << time_limit << "," << dm_blocks << ","
             << dm_edges << "," << dm_peak_mb << "," << build_secs << "," << build_peak_mb << ","
             << simplices << "\n";
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
        jout << "\"build_secs\":" << build_secs << ",\"build_peak_mb\":" << build_peak_mb << ",\"simplices\":" << simplices << "}\\n";
    }

    return 0;
}
