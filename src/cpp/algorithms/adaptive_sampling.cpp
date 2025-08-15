#include "tda/algorithms/adaptive_sampling.hpp"
#include <algorithm>
#include <random>
#include <set>
#include <unordered_set>
#include <queue>
#include <map>
#include <cmath>
#include <chrono>
#include <numeric>
#include <cassert>
#include <functional>

namespace tda::algorithms {

tda::core::Result<AdaptiveSampling::Result> AdaptiveSampling::adaptiveSample(
    const std::vector<std::vector<double>>& points,
    const SamplingConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Validate input
    if (points.empty()) {
        return tda::core::Result<Result>::failure("Empty point cloud");
    }
    
    if (config.min_samples > config.max_samples) {
        return tda::core::Result<Result>::failure("Minimum samples cannot exceed maximum samples");
    }
    
    try {
        Result result;
        std::string strategy_name = config.strategy;
        
        // Route to appropriate sampling strategy
        if (config.strategy == "density") {
            result = densityBasedSampling(points, config);
        } else if (config.strategy == "geometric") {
            result = geometricSampling(points, config);
        } else if (config.strategy == "curvature") {
            result = curvatureBasedSampling(points, config);
        } else if (config.strategy == "hybrid") {
            result = hybridSampling(points, config);
        } else {
            // Default to density-based sampling
            result = densityBasedSampling(points, config);
            strategy_name = "density";
        }
        
        // Preserve strategy information after method call
        result.strategy_used = strategy_name;
        
        // Compute final metrics
        result.achieved_quality = evaluateSamplingQuality(points, result.selected_indices, config.coverage_radius);
        result.coverage_efficiency = computeCoverageEfficiency(points, result.selected_indices, config.coverage_radius);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.computation_time_seconds = duration.count() / 1000.0;
        
        return tda::core::Result<Result>::success(std::move(result));
        
    } catch (const std::exception& e) {
        return tda::core::Result<Result>::failure(
            std::string("Adaptive sampling failed: ") + e.what());
    }
}

tda::core::Result<AdaptiveSampling::Result> AdaptiveSampling::adaptiveSample(
    const std::vector<std::vector<double>>& points
) {
    return adaptiveSample(points, config_);
}

AdaptiveSampling::Result AdaptiveSampling::densityBasedSampling(
    const std::vector<std::vector<double>>& points,
    const SamplingConfig& config
) {
    Result result;
    
    // Compute local densities
    double radius = config.adaptive_radius ? 
        config.coverage_radius : config.coverage_radius;
    result.local_densities = computeLocalDensities(points, radius);
    
    // Detect boundary points if required
    if (config.preserve_boundary) {
        result.boundary_points = detectBoundaryPoints(points, result.local_densities, config.boundary_threshold);
    } else {
        result.boundary_points.resize(points.size(), false);
    }
    
    // Compute sampling weights (inversely proportional to density)
    std::vector<double> empty_curvatures; // Not used for density-based sampling
    result.sampling_weights = computeSamplingWeights(
        points, result.local_densities, result.boundary_points, 
        empty_curvatures, "density"
    );
    
    // Determine target sample size
    size_t target_samples = std::min(config.max_samples, 
        std::max(config.min_samples, 
            static_cast<size_t>(points.size() * config.density_threshold)));
    
    // Select points based on weights
    result.selected_indices = selectPointsByWeight(
        result.sampling_weights, target_samples, config.min_samples, config.max_samples
    );
    
    return result;
}

AdaptiveSampling::Result AdaptiveSampling::geometricSampling(
    const std::vector<std::vector<double>>& points,
    const SamplingConfig& config
) {
    Result result;
    
    // Initialize empty analysis data
    result.local_densities.resize(points.size(), 1.0);
    result.boundary_points.resize(points.size(), false);
    result.sampling_weights.resize(points.size(), 1.0);
    
    // Use Poisson disk sampling for geometric distribution
    result.selected_indices = poissonDiskSampling(points, config.coverage_radius, config.max_samples);
    
    // Ensure minimum samples
    if (result.selected_indices.size() < config.min_samples) {
        // Fill remaining samples randomly
        std::vector<size_t> remaining_indices;
        std::set<size_t> selected_set(result.selected_indices.begin(), result.selected_indices.end());
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (selected_set.find(i) == selected_set.end()) {
                remaining_indices.push_back(i);
            }
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(remaining_indices.begin(), remaining_indices.end(), gen);
        
        size_t additional_needed = config.min_samples - result.selected_indices.size();
        for (size_t i = 0; i < additional_needed && i < remaining_indices.size(); ++i) {
            result.selected_indices.push_back(remaining_indices[i]);
        }
    }
    
    return result;
}

AdaptiveSampling::Result AdaptiveSampling::curvatureBasedSampling(
    const std::vector<std::vector<double>>& points,
    const SamplingConfig& config
) {
    Result result;
    
    // Compute local densities and curvatures
    result.local_densities = computeLocalDensities(points, config.coverage_radius);
    result.curvature_estimates = estimateLocalCurvature(points, config.neighborhood_size);
    
    // Detect boundary points
    if (config.preserve_boundary) {
        result.boundary_points = detectBoundaryPoints(points, result.local_densities, config.boundary_threshold);
    } else {
        result.boundary_points.resize(points.size(), false);
    }
    
    // Compute sampling weights (higher weight for high curvature areas)
    result.sampling_weights = computeSamplingWeights(
        points, result.local_densities, result.boundary_points, 
        result.curvature_estimates, "curvature"
    );
    
    // Determine target sample size
    size_t target_samples = std::min(config.max_samples, 
        std::max(config.min_samples, 
            static_cast<size_t>(points.size() * config.density_threshold)));
    
    // Select points based on weights
    result.selected_indices = selectPointsByWeight(
        result.sampling_weights, target_samples, config.min_samples, config.max_samples
    );
    
    return result;
}

AdaptiveSampling::Result AdaptiveSampling::hybridSampling(
    const std::vector<std::vector<double>>& points,
    const SamplingConfig& config
) {
    Result result;
    
    // Compute all analysis data
    result.local_densities = computeLocalDensities(points, config.coverage_radius);
    result.curvature_estimates = estimateLocalCurvature(points, config.neighborhood_size);
    
    if (config.preserve_boundary) {
        result.boundary_points = detectBoundaryPoints(points, result.local_densities, config.boundary_threshold);
    } else {
        result.boundary_points.resize(points.size(), false);
    }
    
    // Compute hybrid sampling weights
    result.sampling_weights = computeSamplingWeights(
        points, result.local_densities, result.boundary_points, 
        result.curvature_estimates, "hybrid"
    );
    
    // Adaptive sampling with quality target
    size_t target_samples = config.min_samples;
    std::vector<size_t> best_selection;
    double best_quality = 0.0;
    result.iterations_performed = 0;
    
    const size_t max_iterations = 10;
    while (target_samples <= config.max_samples && result.iterations_performed < max_iterations) {
        auto selection = selectPointsByWeight(
            result.sampling_weights, target_samples, config.min_samples, config.max_samples
        );
        
        double quality = evaluateSamplingQuality(points, selection, config.coverage_radius);
        
        if (quality > best_quality) {
            best_quality = quality;
            best_selection = selection;
        }
        
        if (quality >= config.quality_target) {
            break;
        }
        
        target_samples = std::min(config.max_samples, 
            static_cast<size_t>(target_samples * 1.2)); // Increase by 20%
        result.iterations_performed++;
    }
    
    result.selected_indices = best_selection;
    result.achieved_quality = best_quality;
    
    return result;
}

std::vector<double> AdaptiveSampling::computeLocalDensities(
    const std::vector<std::vector<double>>& points,
    double radius
) {
    std::vector<double> densities(points.size(), 0.0);
    
    for (size_t i = 0; i < points.size(); ++i) {
        auto neighbors = findNeighborsInRadius(points, i, radius);
        densities[i] = static_cast<double>(neighbors.size()) / (points.size() * radius * radius);
    }
    
    return densities;
}

std::vector<bool> AdaptiveSampling::detectBoundaryPoints(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& densities,
    double threshold
) {
    std::vector<bool> boundary_points(points.size(), false);
    
    if (densities.empty()) return boundary_points;
    
    // Compute density statistics
    double mean_density = std::accumulate(densities.begin(), densities.end(), 0.0) / densities.size();
    
    for (size_t i = 0; i < points.size(); ++i) {
        // Points with density below threshold relative to mean are considered boundary
        if (densities[i] < threshold * mean_density) {
            boundary_points[i] = true;
        }
    }
    
    return boundary_points;
}

std::vector<double> AdaptiveSampling::estimateLocalCurvature(
    const std::vector<std::vector<double>>& points,
    size_t neighborhood_size
) {
    std::vector<double> curvatures(points.size(), 0.0);
    
    for (size_t i = 0; i < points.size(); ++i) {
        auto neighbors = findKNearestNeighbors(points, i, neighborhood_size);
        
        if (neighbors.size() < 3) {
            curvatures[i] = 0.0;
            continue;
        }
        
        // Simple curvature estimation based on deviation from local plane
        // Compute centroid of neighborhood
        std::vector<double> centroid(points[i].size(), 0.0);
        for (size_t neighbor_idx : neighbors) {
            for (size_t d = 0; d < points[i].size(); ++d) {
                centroid[d] += points[neighbor_idx][d];
            }
        }
        for (size_t d = 0; d < centroid.size(); ++d) {
            centroid[d] /= neighbors.size();
        }
        
        // Compute average distance to centroid (simple curvature proxy)
        double total_deviation = 0.0;
        for (size_t neighbor_idx : neighbors) {
            total_deviation += euclideanDistance(points[neighbor_idx], centroid);
        }
        
        curvatures[i] = total_deviation / neighbors.size();
    }
    
    return curvatures;
}

std::vector<double> AdaptiveSampling::computeSamplingWeights(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& densities,
    const std::vector<bool>& boundary_points,
    const std::vector<double>& curvatures,
    const std::string& strategy
) {
    std::vector<double> weights(points.size(), 1.0);
    
    if (strategy == "density") {
        // Inverse density weighting
        for (size_t i = 0; i < points.size(); ++i) {
            weights[i] = 1.0 / (densities[i] + 1e-8);  // Add small epsilon to avoid division by zero
            if (boundary_points[i]) {
                weights[i] *= 2.0;  // Boost boundary points
            }
        }
    } else if (strategy == "curvature") {
        // High curvature weighting
        for (size_t i = 0; i < points.size(); ++i) {
            weights[i] = curvatures[i] + 0.1;  // Base weight plus curvature
            if (boundary_points[i]) {
                weights[i] *= 1.5;  // Boost boundary points
            }
        }
    } else if (strategy == "hybrid") {
        // Combine density and curvature
        for (size_t i = 0; i < points.size(); ++i) {
            double density_weight = 1.0 / (densities[i] + 1e-8);
            double curvature_weight = curvatures.empty() ? 1.0 : (curvatures[i] + 0.1);
            weights[i] = 0.5 * density_weight + 0.5 * curvature_weight;
            
            if (boundary_points[i]) {
                weights[i] *= 1.8;  // Strong boost for boundary points
            }
        }
    }
    
    return weights;
}

std::vector<size_t> AdaptiveSampling::selectPointsByWeight(
    const std::vector<double>& weights,
    size_t target_samples,
    size_t min_samples,
    size_t max_samples
) {
    // Clamp target samples to valid range
    target_samples = std::max(min_samples, std::min(max_samples, target_samples));
    target_samples = std::min(target_samples, weights.size());
    
    // Create weighted indices
    std::vector<std::pair<double, size_t>> weighted_indices;
    for (size_t i = 0; i < weights.size(); ++i) {
        weighted_indices.push_back({weights[i], i});
    }
    
    // Sort by weight (descending)
    std::sort(weighted_indices.begin(), weighted_indices.end(), 
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top weighted points
    std::vector<size_t> selected_indices;
    selected_indices.reserve(target_samples);
    
    for (size_t i = 0; i < target_samples; ++i) {
        selected_indices.push_back(weighted_indices[i].second);
    }
    
    return selected_indices;
}

std::vector<size_t> AdaptiveSampling::poissonDiskSampling(
    const std::vector<std::vector<double>>& points,
    double radius,
    size_t max_samples
) {
    std::vector<size_t> selected_indices;
    std::set<size_t> selected_set;
    
    if (points.empty()) return selected_indices;
    
    // Start with a random point
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
    
    size_t first_point = dist(gen);
    selected_indices.push_back(first_point);
    selected_set.insert(first_point);
    
    // Iteratively add points that are at least 'radius' away from all selected points
    while (selected_indices.size() < max_samples) {
        bool found_valid_point = false;
        size_t attempts = 0;
        const size_t max_attempts = points.size();
        
        while (attempts < max_attempts && !found_valid_point) {
            size_t candidate = dist(gen);
            
            if (selected_set.find(candidate) != selected_set.end()) {
                attempts++;
                continue;
            }
            
            bool valid = true;
            for (size_t selected_idx : selected_indices) {
                if (euclideanDistance(points[candidate], points[selected_idx]) < radius) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                selected_indices.push_back(candidate);
                selected_set.insert(candidate);
                found_valid_point = true;
            }
            
            attempts++;
        }
        
        if (!found_valid_point) {
            break;  // No more valid points can be found
        }
    }
    
    return selected_indices;
}

double AdaptiveSampling::evaluateSamplingQuality(
    const std::vector<std::vector<double>>& points,
    const std::vector<size_t>& selected_indices,
    double coverage_radius
) {
    if (points.empty() || selected_indices.empty()) return 0.0;
    
    // Count how many original points are within coverage_radius of any selected point
    size_t covered_points = 0;
    
    for (size_t i = 0; i < points.size(); ++i) {
        bool is_covered = false;
        for (size_t selected_idx : selected_indices) {
            if (euclideanDistance(points[i], points[selected_idx]) <= coverage_radius) {
                is_covered = true;
                break;
            }
        }
        if (is_covered) {
            covered_points++;
        }
    }
    
    return static_cast<double>(covered_points) / points.size();
}

double AdaptiveSampling::computeCoverageEfficiency(
    const std::vector<std::vector<double>>& points,
    const std::vector<size_t>& selected_indices,
    double coverage_radius
) {
    if (selected_indices.empty()) return 0.0;
    
    double coverage_quality = evaluateSamplingQuality(points, selected_indices, coverage_radius);
    double sampling_ratio = static_cast<double>(selected_indices.size()) / points.size();
    
    // Efficiency = coverage / sampling_ratio (higher coverage with fewer samples is better)
    return sampling_ratio > 0 ? coverage_quality / sampling_ratio : 0.0;
}

double AdaptiveSampling::euclideanDistance(
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

std::vector<size_t> AdaptiveSampling::findKNearestNeighbors(
    const std::vector<std::vector<double>>& points,
    size_t query_idx,
    size_t k
) {
    if (query_idx >= points.size()) return {};
    
    std::vector<std::pair<double, size_t>> distances;
    distances.reserve(points.size() - 1);
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (i != query_idx) {
            double dist = euclideanDistance(points[query_idx], points[i]);
            distances.push_back({dist, i});
        }
    }
    
    // Sort by distance and take k nearest
    std::sort(distances.begin(), distances.end());
    
    std::vector<size_t> neighbors;
    neighbors.reserve(std::min(k, distances.size()));
    
    for (size_t i = 0; i < std::min(k, distances.size()); ++i) {
        neighbors.push_back(distances[i].second);
    }
    
    return neighbors;
}

std::vector<size_t> AdaptiveSampling::findNeighborsInRadius(
    const std::vector<std::vector<double>>& points,
    size_t query_idx,
    double radius
) {
    if (query_idx >= points.size()) return {};
    
    std::vector<size_t> neighbors;
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (i != query_idx) {
            double dist = euclideanDistance(points[query_idx], points[i]);
            if (dist <= radius) {
                neighbors.push_back(i);
            }
        }
    }
    
    return neighbors;
}

} // namespace tda::algorithms
