#include "tda/algorithms/sparse_rips.hpp"
#include <algorithm>
#include <random>
#include <set>
#include <unordered_set>
#include <queue>
#include <map>
#include <cmath>
#include <chrono>
#include <numeric>
#include <iostream>

namespace tda::algorithms {

tda::core::Result<SparseRips::Result> SparseRips::computeApproximation(
    const std::vector<std::vector<double>>& points,
    double threshold,
    const Config& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Validate input
    if (points.empty()) {
        return tda::core::Result<Result>::failure("Empty point cloud");
    }
    
    if (threshold <= 0.0) {
        return tda::core::Result<Result>::failure("Threshold must be positive");
    }
    
    // Use provided config 
    Config working_config = config;
    
    SparseRips::Result result;
    result.total_edges_considered = 0;
    result.edges_retained = 0;
    
    try {
        // For small point clouds, use exact computation
        if (points.size() < working_config.min_points_threshold) {
            return computeExactForSmallDataset(points, threshold, working_config);
        }
        
        // Step 1: Select landmarks if enabled
        std::vector<size_t> landmarks;
        if (working_config.use_landmarks && points.size() > working_config.num_landmarks) {
            landmarks = selectLandmarks(points, working_config.num_landmarks);
            result.selected_landmarks = landmarks;
        } else {
            // No landmarks - empty list
            landmarks.clear();
            result.selected_landmarks.clear();
        }
        
        // Step 2: Select sparse edges
        auto edges = selectSparseEdges(points, threshold, working_config.sparsity_factor);
        
        // The selectSparseEdges function already handles the sparsity calculation
        // We need to calculate the total edges that were considered (within threshold)
        size_t total_possible_edges = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                double dist = euclideanDistance(points[i], points[j]);
                if (dist <= threshold) {
                    total_possible_edges++;
                }
            }
        }
        
        result.total_edges_considered = total_possible_edges;
        result.edges_retained = edges.size();
        
        // Check edge limit
        if (edges.size() > working_config.max_edges) {
            // Sort by distance and keep only the shortest edges
            std::sort(edges.begin(), edges.end(), 
                [](const auto& a, const auto& b) { return a.second < b.second; });
            edges.resize(working_config.max_edges);
            result.edges_retained = working_config.max_edges;
        }
        
        // Step 3: Build filtration from selected edges
        auto [simplices, filtration_values] = buildFiltrationFromEdges(edges, working_config.max_dimension);
        result.simplices = std::move(simplices);
        result.filtration_values = std::move(filtration_values);
        
        // Step 4: Estimate approximation quality
        result.approximation_quality = estimateApproximationQuality(
            result.total_edges_considered, result.edges_retained, landmarks);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.computation_time_seconds = duration.count() / 1000.0;
        
        return tda::core::Result<Result>::success(std::move(result));
        
    } catch (const std::exception& e) {
        return tda::core::Result<Result>::failure(
            std::string("Sparse Rips computation failed: ") + e.what());
    }
}

tda::core::Result<SparseRips::Result> SparseRips::computeApproximation(
    const std::vector<std::vector<double>>& points,
    double threshold
) {
    return computeApproximation(points, threshold, config_);
}

std::vector<size_t> SparseRips::selectLandmarks(
    const std::vector<std::vector<double>>& points,
    size_t num_landmarks
) {
    if (num_landmarks >= points.size()) {
        std::vector<size_t> all_points(points.size());
        std::iota(all_points.begin(), all_points.end(), 0);
        return all_points;
    }
    
    std::vector<size_t> landmarks;
    landmarks.reserve(num_landmarks);
    
    // Farthest point sampling
    std::vector<double> min_distances(points.size(), std::numeric_limits<double>::max());
    
    // Start with a random point
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
    size_t first_landmark = dist(gen);
    landmarks.push_back(first_landmark);
    
    // Update distances to first landmark
    for (size_t i = 0; i < points.size(); ++i) {
        if (i != first_landmark) {
            min_distances[i] = euclideanDistance(points[i], points[first_landmark]);
        }
    }
    
    // Select remaining landmarks
    for (size_t l = 1; l < num_landmarks; ++l) {
        // Find point with maximum distance to nearest landmark
        size_t farthest_point = 0;
        double max_distance = 0.0;
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (min_distances[i] > max_distance) {
                max_distance = min_distances[i];
                farthest_point = i;
            }
        }
        
        landmarks.push_back(farthest_point);
        min_distances[farthest_point] = 0.0;  // Mark as landmark
        
        // Update distances to new landmark
        for (size_t i = 0; i < points.size(); ++i) {
            if (min_distances[i] > 0.0) {
                double dist = euclideanDistance(points[i], points[farthest_point]);
                min_distances[i] = std::min(min_distances[i], dist);
            }
        }
    }
    
    return landmarks;
}

std::vector<std::pair<std::pair<size_t, size_t>, double>> SparseRips::selectSparseEdges(
    const std::vector<std::vector<double>>& points,
    double threshold,
    double sparsity_factor
) {
    if (config_.strategy == "density") {
        return densityBasedSelection(points, threshold, sparsity_factor);
    } else if (config_.strategy == "geometric") {
        return geometricSelection(points, threshold, sparsity_factor);
    } else {
        // Hybrid approach - use both strategies and combine
        auto density_edges = densityBasedSelection(points, threshold, sparsity_factor * 0.7);
        auto geometric_edges = geometricSelection(points, threshold, sparsity_factor * 0.3);
        
        // Combine and remove duplicates
        std::set<std::pair<size_t, size_t>> edge_set;
        std::vector<std::pair<std::pair<size_t, size_t>, double>> combined_edges;
        
        for (const auto& edge : density_edges) {
            auto [i, j] = edge.first;
            if (edge_set.find({std::min(i, j), std::max(i, j)}) == edge_set.end()) {
                edge_set.insert({std::min(i, j), std::max(i, j)});
                combined_edges.push_back(edge);
            }
        }
        
        for (const auto& edge : geometric_edges) {
            auto [i, j] = edge.first;
            if (edge_set.find({std::min(i, j), std::max(i, j)}) == edge_set.end()) {
                edge_set.insert({std::min(i, j), std::max(i, j)});
                combined_edges.push_back(edge);
            }
        }
        
        return combined_edges;
    }
}

std::vector<std::pair<std::pair<size_t, size_t>, double>> SparseRips::densityBasedSelection(
    const std::vector<std::vector<double>>& points,
    double threshold,
    double sparsity_factor
) {
    std::vector<std::pair<std::pair<size_t, size_t>, double>> all_edges;
    
    // Compute all edges within threshold
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dist = euclideanDistance(points[i], points[j]);
            if (dist <= threshold) {
                all_edges.push_back({{i, j}, dist});
            }
        }
    }
    
    // Sort by distance (shorter edges first)
    std::sort(all_edges.begin(), all_edges.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Keep only the fraction specified by sparsity_factor
    size_t edges_to_keep = static_cast<size_t>(all_edges.size() * sparsity_factor);
    if (edges_to_keep > all_edges.size()) {
        edges_to_keep = all_edges.size();
    }
    
    all_edges.resize(edges_to_keep);
    return all_edges;
}

std::vector<std::pair<std::pair<size_t, size_t>, double>> SparseRips::geometricSelection(
    const std::vector<std::vector<double>>& points,
    double threshold,
    double sparsity_factor
) {
    std::vector<std::pair<std::pair<size_t, size_t>, double>> edges;
    
    // Geometric sparsification: use a form of Well-Separated Pair Decomposition approximation
    double grid_size = threshold * 0.5;  // Grid size for geometric partitioning
    
    // Simple geometric approach: divide space into grid and sample edges more densely
    // from regions with fewer points
    std::map<std::pair<int, int>, std::vector<size_t>> grid;
    
    // Find bounding box
    if (points.empty() || points[0].empty()) return edges;
    
    double min_x = points[0][0], max_x = points[0][0];
    double min_y = points[0].size() > 1 ? points[0][1] : 0.0;
    double max_y = min_y;
    
    for (const auto& point : points) {
        if (!point.empty()) {
            min_x = std::min(min_x, point[0]);
            max_x = std::max(max_x, point[0]);
            if (point.size() > 1) {
                min_y = std::min(min_y, point[1]);
                max_y = std::max(max_y, point[1]);
            }
        }
    }
    
    // Assign points to grid cells
    for (size_t i = 0; i < points.size(); ++i) {
        if (!points[i].empty()) {
            int grid_x = static_cast<int>((points[i][0] - min_x) / grid_size);
            int grid_y = points[i].size() > 1 ? 
                static_cast<int>((points[i][1] - min_y) / grid_size) : 0;
            grid[{grid_x, grid_y}].push_back(i);
        }
    }
    
    // Sample edges with bias toward sparser regions
    for (const auto& [cell, point_indices] : grid) {
        // Higher sampling rate for sparser cells
        double local_sparsity = std::min(1.0, sparsity_factor * (10.0 / point_indices.size()));
        
        for (size_t i = 0; i < point_indices.size(); ++i) {
            for (size_t j = i + 1; j < point_indices.size(); ++j) {
                size_t pi = point_indices[i];
                size_t pj = point_indices[j];
                double dist = euclideanDistance(points[pi], points[pj]);
                
                if (dist <= threshold) {
                    // Sample based on local sparsity
                    static std::random_device rd;
                    static std::mt19937 gen(rd());
                    std::uniform_real_distribution<double> uniform(0.0, 1.0);
                    
                    if (uniform(gen) < local_sparsity) {
                        edges.push_back({{pi, pj}, dist});
                    }
                }
            }
        }
    }
    
    return edges;
}

double SparseRips::euclideanDistance(
    const std::vector<double>& p1,
    const std::vector<double>& p2
) const {
    if (p1.size() != p2.size()) {
        throw std::invalid_argument("Points must have same dimension");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::pair<std::vector<std::vector<size_t>>, std::vector<double>>
SparseRips::buildFiltrationFromEdges(
    const std::vector<std::pair<std::pair<size_t, size_t>, double>>& edges,
    int max_dimension
) {
    std::vector<std::vector<size_t>> simplices;
    std::vector<double> filtration_values;
    
    // Add vertices (0-simplices)
    std::set<size_t> vertices;
    for (const auto& edge : edges) {
        vertices.insert(edge.first.first);
        vertices.insert(edge.first.second);
    }
    
    for (size_t vertex : vertices) {
        simplices.push_back({vertex});
        filtration_values.push_back(0.0);  // Vertices appear at filtration value 0
    }
    
    // Add edges (1-simplices)
    for (const auto& edge : edges) {
        simplices.push_back({edge.first.first, edge.first.second});
        filtration_values.push_back(edge.second);
    }
    
    // For higher dimensions, we would need more complex algorithms
    // For now, only implement up to 1-skeleton (edges)
    if (max_dimension > 1) {
        // TODO: Implement triangle and higher-dimensional simplex detection
        // This would require checking if three or more points form valid simplices
        // based on the edge structure
    }
    
    return {simplices, filtration_values};
}

double SparseRips::estimateApproximationQuality(
    size_t original_edges,
    size_t retained_edges,
    const std::vector<size_t>& landmarks
) const {
    if (original_edges == 0) return 1.0;
    
    // Basic quality estimate based on edge retention and landmark coverage
    double edge_ratio = static_cast<double>(retained_edges) / original_edges;
    double landmark_ratio = landmarks.empty() ? 1.0 : 
        std::min(1.0, static_cast<double>(landmarks.size()) / 1000.0);  // Normalize to expected landmark count
    
    // Combined quality metric (weighted average)
    return 0.7 * edge_ratio + 0.3 * landmark_ratio;
}

tda::core::Result<SparseRips::Result> SparseRips::computeExactForSmallDataset(
    const std::vector<std::vector<double>>& points,
    double threshold,
    const Config& config
) {
    // For small datasets, compute exact Vietoris-Rips
    SparseRips::Result result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::pair<std::pair<size_t, size_t>, double>> all_edges;
    
    // Compute all edges
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dist = euclideanDistance(points[i], points[j]);
            if (dist <= threshold) {
                all_edges.push_back({{i, j}, dist});
            }
        }
    }
    
    result.total_edges_considered = all_edges.size();
    result.edges_retained = all_edges.size();
    
    // Build complete filtration
    auto [simplices, filtration_values] = buildFiltrationFromEdges(all_edges, config.max_dimension);
    result.simplices = std::move(simplices);
    result.filtration_values = std::move(filtration_values);
    
    // All points are landmarks in exact computation
    result.selected_landmarks.resize(points.size());
    std::iota(result.selected_landmarks.begin(), result.selected_landmarks.end(), 0);
    
    result.approximation_quality = 1.0;  // Exact computation
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.computation_time_seconds = duration.count() / 1000.0;
    
    return tda::core::Result<Result>::success(std::move(result));
}

} // namespace tda::algorithms
