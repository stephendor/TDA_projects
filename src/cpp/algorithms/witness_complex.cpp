#include "tda/algorithms/witness_complex.hpp"
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

tda::core::Result<WitnessComplex::Result> WitnessComplex::computeWitnessComplex(
    const std::vector<std::vector<double>>& points,
    const WitnessConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Validate input
    if (points.empty()) {
        return tda::core::Result<Result>::failure("Empty point cloud");
    }
    
    if (config.num_landmarks == 0) {
        return tda::core::Result<Result>::failure("Number of landmarks must be positive");
    }
    
    if (config.max_dimension < 0) {
        return tda::core::Result<Result>::failure("Maximum dimension must be non-negative");
    }
    
    WitnessComplex::Result result;
    result.total_witness_checks = 0;
    result.strategy_used = config.landmark_strategy;
    
    try {
        // Step 1: Select landmark points
        std::vector<size_t> landmarks = selectLandmarks(
            points, 
            std::min(config.num_landmarks, points.size()), 
            config.landmark_strategy
        );
        result.landmark_indices = landmarks;
        
        // Step 2: Build witness relations
        auto witness_relations = buildWitnessRelations(points, landmarks, config);
        
        // Step 3: Generate simplices from witness relations
        auto [simplices, filtration_values] = generateSimplicesFromWitnesses(
            landmarks, witness_relations, config.max_dimension, config
        );
        
        result.simplices = std::move(simplices);
        result.filtration_values = std::move(filtration_values);
        
        // Step 4: Store witness relations for analysis
        for (size_t i = 0; i < witness_relations.size(); ++i) {
            std::vector<size_t> witnesses_for_point;
            for (const auto& [landmark_idx, distance] : witness_relations[i]) {
                if (distance <= config.distance_threshold) {
                    witnesses_for_point.push_back(landmark_idx);
                }
            }
            if (!witnesses_for_point.empty()) {
                result.witness_relations.push_back(std::move(witnesses_for_point));
            }
        }
        
        // Step 5: Estimate approximation quality
        result.approximation_quality = estimateApproximationQuality(
            points.size(), landmarks.size(), result.simplices.size(), config.landmark_strategy
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.computation_time_seconds = duration.count() / 1000.0;
        
        return tda::core::Result<Result>::success(std::move(result));
        
    } catch (const std::exception& e) {
        return tda::core::Result<Result>::failure(
            std::string("Witness complex computation failed: ") + e.what());
    }
}

tda::core::Result<WitnessComplex::Result> WitnessComplex::computeWitnessComplex(
    const std::vector<std::vector<double>>& points
) {
    return computeWitnessComplex(points, config_);
}

std::vector<size_t> WitnessComplex::selectLandmarks(
    const std::vector<std::vector<double>>& points,
    size_t num_landmarks,
    const std::string& strategy
) {
    if (num_landmarks >= points.size()) {
        std::vector<size_t> all_points(points.size());
        std::iota(all_points.begin(), all_points.end(), 0);
        return all_points;
    }
    
    if (strategy == "farthest_point") {
        return farthestPointSampling(points, num_landmarks);
    } else if (strategy == "random") {
        return randomSampling(points, num_landmarks);
    } else if (strategy == "density") {
        return densityBasedSampling(points, num_landmarks);
    } else {
        // Default to farthest point sampling
        return farthestPointSampling(points, num_landmarks);
    }
}

std::vector<size_t> WitnessComplex::farthestPointSampling(
    const std::vector<std::vector<double>>& points,
    size_t num_landmarks
) {
    std::vector<size_t> landmarks;
    landmarks.reserve(num_landmarks);
    
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
        } else {
            min_distances[i] = 0.0;
        }
    }
    
    // Select remaining landmarks using farthest point sampling
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

std::vector<size_t> WitnessComplex::randomSampling(
    const std::vector<std::vector<double>>& points,
    size_t num_landmarks
) {
    std::vector<size_t> all_indices(points.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), gen);
    
    all_indices.resize(num_landmarks);
    return all_indices;
}

std::vector<size_t> WitnessComplex::densityBasedSampling(
    const std::vector<std::vector<double>>& points,
    size_t num_landmarks
) {
    // Compute local density for each point
    std::vector<double> densities(points.size(), 0.0);
    const double radius = 1.0;  // Fixed radius for density estimation
    
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < points.size(); ++j) {
            if (i != j) {
                double dist = euclideanDistance(points[i], points[j]);
                if (dist <= radius) {
                    densities[i] += 1.0;
                }
            }
        }
    }
    
    // Select landmarks with probability inversely proportional to density
    std::vector<std::pair<double, size_t>> density_indices;
    for (size_t i = 0; i < points.size(); ++i) {
        double weight = 1.0 / (1.0 + densities[i]);  // Lower density = higher weight
        density_indices.push_back({weight, i});
    }
    
    // Sort by weight (descending)
    std::sort(density_indices.begin(), density_indices.end(), 
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<size_t> landmarks;
    landmarks.reserve(num_landmarks);
    for (size_t i = 0; i < num_landmarks && i < density_indices.size(); ++i) {
        landmarks.push_back(density_indices[i].second);
    }
    
    return landmarks;
}

std::vector<std::vector<std::pair<size_t, double>>> WitnessComplex::buildWitnessRelations(
    const std::vector<std::vector<double>>& points,
    const std::vector<size_t>& landmarks,
    const WitnessConfig& config
) {
    std::vector<std::vector<std::pair<size_t, double>>> witness_relations(points.size());
    
    // For each point, compute distances to all landmarks and sort
    for (size_t i = 0; i < points.size(); ++i) {
        std::vector<std::pair<double, size_t>> distances_to_landmarks;
        distances_to_landmarks.reserve(landmarks.size());
        
        for (size_t j = 0; j < landmarks.size(); ++j) {
            double dist = euclideanDistance(points[i], points[landmarks[j]]);
            distances_to_landmarks.push_back({dist, j});  // j is the landmark index
        }
        
        // Sort by distance
        std::sort(distances_to_landmarks.begin(), distances_to_landmarks.end());
        
        // Store only the closest witnesses (limited by max_witnesses_per_point)
        size_t max_witnesses = std::min(distances_to_landmarks.size(), config.max_witnesses_per_point);
        for (size_t w = 0; w < max_witnesses; ++w) {
            const auto& [dist, landmark_idx] = distances_to_landmarks[w];
            double relaxed_distance = dist + config.relaxation;
            witness_relations[i].push_back({landmark_idx, relaxed_distance});
        }
    }
    
    return witness_relations;
}

bool WitnessComplex::hasSufficientWitnesses(
    const std::vector<size_t>& simplex,
    const std::vector<std::vector<std::pair<size_t, double>>>& witness_relations,
    bool use_strong_witness,
    size_t min_witnesses
) {
    size_t witness_count = 0;
    
    // Check each point as a potential witness
    for (const auto& point_witnesses : witness_relations) {
        bool is_witness = false;
        
        if (use_strong_witness) {
            // Strong witness: all simplex vertices must be among closest landmarks
            std::set<size_t> closest_landmarks;
            for (size_t k = 0; k < simplex.size() && k < point_witnesses.size(); ++k) {
                closest_landmarks.insert(point_witnesses[k].first);
            }
            
            bool all_in_closest = true;
            for (size_t vertex : simplex) {
                if (closest_landmarks.find(vertex) == closest_landmarks.end()) {
                    all_in_closest = false;
                    break;
                }
            }
            is_witness = all_in_closest;
        } else {
            // Weak witness: check if simplex vertices are witnessed with relaxation
            double max_simplex_distance = 0.0;
            bool all_vertices_witnessed = true;
            
            for (size_t vertex : simplex) {
                bool vertex_found = false;
                for (const auto& [landmark_idx, distance] : point_witnesses) {
                    if (landmark_idx == vertex) {
                        max_simplex_distance = std::max(max_simplex_distance, distance);
                        vertex_found = true;
                        break;
                    }
                }
                if (!vertex_found) {
                    all_vertices_witnessed = false;
                    break;
                }
            }
            
            // Check if there's no closer landmark
            if (all_vertices_witnessed && !point_witnesses.empty()) {
                double closest_distance = point_witnesses[0].second;
                is_witness = (max_simplex_distance <= closest_distance);
            }
        }
        
        if (is_witness) {
            witness_count++;
        }
    }
    
    return witness_count >= min_witnesses;
}

std::pair<std::vector<std::vector<size_t>>, std::vector<double>>
WitnessComplex::generateSimplicesFromWitnesses(
    const std::vector<size_t>& landmarks,
    const std::vector<std::vector<std::pair<size_t, double>>>& witness_relations,
    int max_dimension,
    const WitnessConfig& config
) {
    std::vector<std::vector<size_t>> simplices;
    std::vector<double> filtration_values;
    
    // Generate all possible simplices up to max_dimension
    for (int dim = 0; dim <= max_dimension; ++dim) {
        std::vector<std::vector<size_t>> dim_simplices = generateCombinations(
            std::vector<size_t>(landmarks.size()), dim + 1
        );
        
        for (const auto& simplex_indices : dim_simplices) {
            // Convert simplex indices to actual landmark indices
            std::vector<size_t> simplex;
            for (size_t idx : simplex_indices) {
                simplex.push_back(idx);  // These are already landmark indices (0 to landmarks.size()-1)
            }
            
            // Check if this simplex has sufficient witnesses
            if (hasSufficientWitnesses(simplex, witness_relations, 
                                    config.use_strong_witness, config.min_witnesses)) {
                simplices.push_back(simplex);
                
                // Compute filtration value
                double filtration_value = computeFiltrationValue(simplex, witness_relations, config.relaxation);
                filtration_values.push_back(filtration_value);
            }
        }
    }
    
    return {simplices, filtration_values};
}

double WitnessComplex::computeFiltrationValue(
    const std::vector<size_t>& simplex,
    const std::vector<std::vector<std::pair<size_t, double>>>& witness_relations,
    double relaxation
) {
    double max_witness_distance = 0.0;
    
    // Find the maximum distance among all witnesses for this simplex
    for (const auto& point_witnesses : witness_relations) {
        double max_simplex_distance = 0.0;
        bool all_vertices_found = true;
        
        for (size_t vertex : simplex) {
            bool found = false;
            for (const auto& [landmark_idx, distance] : point_witnesses) {
                if (landmark_idx == vertex) {
                    max_simplex_distance = std::max(max_simplex_distance, distance - relaxation);
                    found = true;
                    break;
                }
            }
            if (!found) {
                all_vertices_found = false;
                break;
            }
        }
        
        if (all_vertices_found) {
            max_witness_distance = std::max(max_witness_distance, max_simplex_distance);
        }
    }
    
    return std::max(0.0, max_witness_distance);
}

double WitnessComplex::euclideanDistance(
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

double WitnessComplex::estimateApproximationQuality(
    size_t num_points,
    size_t num_landmarks,
    size_t num_simplices,
    const std::string& strategy
) const {
    if (num_points == 0) return 1.0;
    
    // Quality metrics based on different factors
    double landmark_ratio = static_cast<double>(num_landmarks) / num_points;
    double simplex_density = num_simplices > 0 ? 
        static_cast<double>(num_simplices) / (num_landmarks * num_landmarks) : 0.0;
    
    // Strategy-based quality adjustment
    double strategy_factor = 1.0;
    if (strategy == "farthest_point") {
        strategy_factor = 0.9;  // Generally good coverage
    } else if (strategy == "density") {
        strategy_factor = 0.85; // Good for non-uniform distributions
    } else if (strategy == "random") {
        strategy_factor = 0.7;  // Less reliable coverage
    }
    
    // Combined quality metric
    double base_quality = std::min(1.0, landmark_ratio * 10.0);  // Normalize landmark ratio
    double simplex_quality = std::min(1.0, simplex_density * 5.0); // Normalize simplex density
    
    return strategy_factor * (0.6 * base_quality + 0.4 * simplex_quality);
}

std::vector<std::vector<size_t>> WitnessComplex::generateAllPossibleSimplices(
    const std::vector<size_t>& landmarks,
    int max_dimension
) {
    std::vector<std::vector<size_t>> all_simplices;
    
    for (int dim = 0; dim <= max_dimension; ++dim) {
        auto dim_simplices = generateCombinations(landmarks, dim + 1);
        all_simplices.insert(all_simplices.end(), dim_simplices.begin(), dim_simplices.end());
    }
    
    return all_simplices;
}

std::vector<std::vector<size_t>> WitnessComplex::generateCombinations(
    const std::vector<size_t>& elements,
    size_t k
) {
    std::vector<std::vector<size_t>> combinations;
    
    if (k == 0) {
        combinations.push_back({});
        return combinations;
    }
    
    if (k > elements.size()) {
        return combinations;
    }
    
    // Generate all k-combinations using recursive approach
    std::function<void(size_t, std::vector<size_t>&)> generate;
    generate = [&](size_t start, std::vector<size_t>& current) {
        if (current.size() == k) {
            combinations.push_back(current);
            return;
        }
        
        for (size_t i = start; i < elements.size(); ++i) {
            current.push_back(i);  // Use index directly for landmark indices
            generate(i + 1, current);
            current.pop_back();
        }
    };
    
    std::vector<size_t> current;
    generate(0, current);
    
    return combinations;
}

} // namespace tda::algorithms
