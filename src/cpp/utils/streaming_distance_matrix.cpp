#include "tda/utils/streaming_distance_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>
#include "tda/core/memory_monitor.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tda::utils {

static inline bool use_threshold(const StreamingDMConfig& cfg) {
    return cfg.max_distance >= 0.0;
}

static inline size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// Simple Euclidean distance.
double StreamingDistanceMatrix::euclidean(const Point& a, const Point& b) {
    double s = 0.0;
    const size_t n = a.size();
    for (size_t k = 0; k < n; ++k) {
        double d = a[k] - b[k];
        s += d * d;
    }
    return std::sqrt(s);
}

StreamingDMStats StreamingDistanceMatrix::process(const PointContainer& points) {
    StreamingDMStats stats{};
    const size_t n = points.size();
    stats.total_points = n;
    if (n == 0) return stats;

    const size_t B = std::max<size_t>(1, config_.block_size);
    const bool symmetric = config_.symmetric;
    const bool threshold_mode = use_threshold(config_);

    auto t0 = std::chrono::high_resolution_clock::now();
    // Telemetry: memory baseline
    stats.memory_before_bytes = tda::core::MemoryMonitor::getCurrentMemoryUsage();
    stats.peak_memory_bytes = stats.memory_before_bytes;

    const size_t nblocks = ceil_div(n, B);
    stats.total_blocks = symmetric ? (nblocks * (nblocks + 1)) / 2 : (nblocks * nblocks);

    for (size_t bi = 0; bi < n; bi += B) {
        const size_t i_end = std::min(bi + B, n);
        const size_t i_len = i_end - bi;

        for (size_t bj = symmetric ? bi : 0; bj < n; bj += B) {
            const size_t j_end = std::min(bj + B, n);
            const size_t j_len = j_end - bj;

            // Optional tile buffer only if block callback is used and not symmetric diagonal upper-tri optimization.
            std::vector<std::vector<double>> tile;
            if (block_cb_ && (!symmetric || bi != bj)) {
                tile.assign(i_len, std::vector<double>(j_len, 0.0));
            }

            // Compute tile
            if (symmetric && bi == bj) {
                // Diagonal block: compute only upper triangle (i < j)
                for (size_t i = bi; i < i_end; ++i) {
                    const size_t j_start = i + 1; // upper triangle only
                    for (size_t j = j_start; j < j_end; ++j) {
                        double d = euclidean(points[i], points[j]);
                        stats.total_pairs++;

                        if (threshold_mode) {
                            if (d <= config_.max_distance && edge_cb_) {
                                edge_cb_(i, j, d);
                                stats.emitted_edges++;
                            }
                        } else if (block_cb_) {
                            tile[i - bi][j - bj] = d;
                            // mirror not stored to save memory; consumer can infer symmetry
                        }
                    }
                }
            } else {
                // Off-diagonal block: full i x j tile
                bool can_parallel_blocks = false;
#ifdef _OPENMP
                can_parallel_blocks = (config_.use_parallel && !threshold_mode && block_cb_ && (i_len * j_len >= 256));
                if (can_parallel_blocks) {
#pragma omp parallel for collapse(2) schedule(static)
                    for (size_t ii = 0; ii < i_len; ++ii) {
                        for (size_t jj = 0; jj < j_len; ++jj) {
                            size_t i = bi + ii;
                            size_t j = bj + jj;
                            double d = euclidean(points[i], points[j]);
                            tile[ii][jj] = d;
                        }
                    }
                    // accumulate pair count
                    stats.total_pairs += i_len * j_len;
                } else
#endif
                {
                    for (size_t i = bi; i < i_end; ++i) {
                        for (size_t j = bj; j < j_end; ++j) {
                            double d = euclidean(points[i], points[j]);
                            stats.total_pairs++;
                            if (threshold_mode) {
                                if (d <= config_.max_distance && edge_cb_) {
                                    edge_cb_(i, j, d);
                                    stats.emitted_edges++;
                                }
                            } else if (block_cb_) {
                                tile[i - bi][j - bj] = d;
                            }
                        }
                    }
                }
            }

        // Update telemetry: track peak after each block
        stats.peak_memory_bytes = std::max(stats.peak_memory_bytes, tda::core::MemoryMonitor::getCurrentMemoryUsage());

        if (!threshold_mode && block_cb_ && (!symmetric || bi != bj)) {
                block_cb_(bi, bj, tile);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    stats.memory_after_bytes = tda::core::MemoryMonitor::getCurrentMemoryUsage();
    return stats;
}

} // namespace tda::utils
