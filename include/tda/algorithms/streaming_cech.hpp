#pragma once

#include "tda/core/types.hpp"
#include "tda/core/simplex_pool.hpp"
#include "tda/utils/streaming_distance_matrix.hpp"
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <vector>
#include <memory>

namespace tda::algorithms {

// A bounded-memory, streaming-friendly Čech-like complex builder.
// Uses StreamingDistanceMatrix to form neighbor adjacency under a global threshold,
// then constructs simplices up to maxDimension, with optional adaptive radius heuristics.
class StreamingCechComplex {
public:
    using Point = std::vector<double>;
    using PointContainer = std::vector<Point>;

    struct Config {
        // Geometric knobs
        double radius = 1.0;               // base radius
        double radiusMultiplier = 1.5;     // for adaptive radius cap
        size_t maxDimension = 3;           // max simplex dimension
        size_t maxNeighbors = 64;          // cap per-vertex neighbors
        bool useAdaptiveRadius = true;     // estimate per-vertex radius from local distances

        // Streaming distance matrix knobs
        tda::utils::StreamingDMConfig dm = {
            /*block_size*/ 64,
            /*symmetric*/ true,
            /*max_distance*/ -1.0, // will be set from radius at runtime
            /*use_parallel*/ true
        };
    };

    StreamingCechComplex() = default;
    explicit StreamingCechComplex(const Config& cfg) : config_(cfg) {}

    // Lifecycle
    tda::core::Result<void> initialize(const PointContainer& points);
    tda::core::Result<void> computeComplex();
    tda::core::Result<void> computePersistence(int coefficientField = 2);

    // Introspection
    tda::core::Result<std::vector<tda::core::SimplexInfo>> getSimplices() const;
    tda::core::Result<tda::core::ComplexStatistics> getStatistics() const;
    const Config& getConfig() const { return config_; }
    void updateConfig(const Config& cfg) { config_ = cfg; }
    void clear();

    // Telemetry from the streaming distance pass
    tda::utils::StreamingDMStats getStreamingStats() const { return dm_stats_; }
    
    // Telemetry for complex construction (this class's computeComplex phase)
    struct BuildStats {
        size_t memory_before_bytes = 0;
        size_t memory_after_bytes = 0;
        size_t peak_memory_bytes = 0;
        double elapsed_seconds = 0.0;
        size_t num_simplices_built = 0;
    // New telemetry
    std::vector<size_t> simplex_count_by_dim; // counts per dimension built during computeComplex
    std::vector<size_t> adjacency_histogram;  // histogram of neighbor list sizes
    size_t adjacency_max_degree = 0;          // max degree observed
    // SimplexPool telemetry
    size_t pool_total_blocks = 0;             // total blocks across all buckets
    size_t pool_free_blocks = 0;              // free blocks across all buckets
    double pool_fragmentation = 0.0;          // 1 - (free/total) when total>0
    std::vector<std::pair<size_t, std::pair<size_t,size_t>>> pool_bucket_stats; // (arity, (total,free))
    size_t pool_pages = 0;                    // total allocated pages across buckets (optional)
    size_t pool_blocks_per_page = 0;          // blocks per page (representative)
    };
    BuildStats getBuildStats() const { return build_stats_; }

private:
    // Types
    using Simplex_tree = Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
    using Filtration_value = double;
    using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Gudhi::persistent_cohomology::Field_Zp>;

    // Config and data
    Config config_{};
    PointContainer points_{};
    std::vector<std::vector<size_t>> neighbors_; // adjacency from streaming edges

    // Pooled temporaries
    tda::core::SimplexPool simplex_pool_{};

    // GUDHI holders
    std::unique_ptr<Simplex_tree> simplex_tree_{};
    std::unique_ptr<Persistent_cohomology> persistent_cohomology_{};

    // Telemetry
    tda::utils::StreamingDMStats dm_stats_{};
    BuildStats build_stats_{};

    // Helpers
    static double euclidean(const Point& a, const Point& b);
    double estimateAdaptiveRadius(size_t idx) const;
    void buildNeighborsStreaming();
    void addSimplexToTree(const std::vector<size_t>& simplex, double filtrationValue);
};

} // namespace tda::algorithms
