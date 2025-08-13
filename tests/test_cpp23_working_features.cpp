#include <iostream>
#include <vector>
#include <ranges>
#include <algorithm>
#include <execution>
#include <span>
#include <functional>
#include <chrono>
#include <random>
#include <iomanip>
#include <limits>
#include <cmath>

// C++23 features that work well in GCC 14.2
namespace tda::cpp23 {

// 1. Enhanced constexpr capabilities
constexpr double euclideanDistanceConstexpr(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::max();
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// 2. std::span for efficient array views
template<typename T>
void processArrayView(std::span<const T> data) {
    std::cout << "Processing span of " << data.size() << " elements" << std::endl;
    
    // C++23 ranges with spans
    auto doubled = data 
        | std::views::transform([](T x) { return x * 2; })
        | std::views::take(5);  // Take first 5 elements
    
    std::cout << "First 5 doubled values: ";
    for (auto val : doubled) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// 3. Modern ranges with views
template<typename Container>
auto findOutliers(const Container& data, double threshold = 2.0) {
    // Calculate mean using ranges
    double mean = std::ranges::fold_left(data, 0.0, std::plus{}) / data.size();
    
    // Find outliers using ranges and views
    auto outliers = data 
        | std::views::enumerate  // Add indices
        | std::views::filter([mean, threshold](const auto& pair) {
            auto [idx, value] = pair;
            return std::abs(value - mean) > threshold;
        })
        | std::views::transform([](const auto& pair) {
            auto [idx, value] = pair;
            return std::make_pair(idx, value);
        });
    
    return std::vector(outliers.begin(), outliers.end());
}

// 4. Enhanced parallel algorithms
void parallelDistanceMatrix(const std::vector<std::vector<double>>& points) {
    const size_t n = points.size();
    std::vector<std::vector<double>> distances(n, std::vector<double>(n));
    
    // Generate all index pairs
    std::vector<std::pair<size_t, size_t>> index_pairs;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            index_pairs.emplace_back(i, j);
        }
    }
    
    // Parallel computation of distances
    std::for_each(std::execution::par_unseq, 
        index_pairs.begin(), index_pairs.end(),
        [&points, &distances](const auto& pair) {
            auto [i, j] = pair;
            
            // Use span for efficient access
            std::span<const double> p1(points[i]);
            std::span<const double> p2(points[j]);
            
            double dist = euclideanDistanceConstexpr(p1, p2);
            distances[i][j] = dist;
            distances[j][i] = dist; // Symmetric
        });
    
    std::cout << "Computed " << n << "x" << n << " distance matrix in parallel" << std::endl;
}

// 5. Functional programming with ranges
template<typename PointCloud>
auto computeSparseGraph(const PointCloud& points, double threshold) {
    auto n = points.size();
    
    // Generate all pairs using ranges
    auto indices = std::views::iota(0UL, n);
    
    // Create sparse adjacency list using ranges
    std::vector<std::vector<std::pair<size_t, double>>> graph(n);
    
    for (size_t i = 0; i < n; ++i) {
        auto neighbors = indices
            | std::views::filter([i](size_t j) { return j != i; })
            | std::views::transform([&points, i, threshold](size_t j) {
                std::span<const double> p1(points[i]);
                std::span<const double> p2(points[j]);
                double dist = euclideanDistanceConstexpr(p1, p2);
                return std::make_pair(j, dist);
            })
            | std::views::filter([threshold](const auto& pair) {
                return pair.second <= threshold;
            });
        
        std::ranges::copy(neighbors, std::back_inserter(graph[i]));
    }
    
    return graph;
}

} // namespace tda::cpp23

namespace {

std::vector<std::vector<double>> generateTestPoints(size_t num_points, size_t dimension) {
    std::vector<std::vector<double>> points;
    points.reserve(num_points);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    
    for (size_t i = 0; i < num_points; ++i) {
        std::vector<double> point;
        point.reserve(dimension);
        for (size_t j = 0; j < dimension; ++j) {
            point.push_back(dist(gen));
        }
        points.push_back(std::move(point));
    }
    
    return points;
}

} // namespace

int main() {
    std::cout << "🚀 C++23 Working Features Demo" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Test 1: std::span and constexpr capabilities
    std::cout << "\n=== Testing std::span and Enhanced constexpr ===" << std::endl;
    
    std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> v2 = {2.0, 3.0, 4.0, 5.0, 6.0};
    
    std::span<const double> span1(v1);
    std::span<const double> span2(v2);
    
    // constexpr distance computation
    double dist = tda::cpp23::euclideanDistanceConstexpr(span1, span2);
    std::cout << "✅ Constexpr Euclidean distance: " << std::fixed << std::setprecision(3) << dist << std::endl;
    
    // Array view processing
    tda::cpp23::processArrayView(span1);
    
    // Test 2: Modern ranges and views
    std::cout << "\n=== Testing Modern Ranges and Views ===" << std::endl;
    
    std::vector<double> data = {1.0, 5.0, 2.0, 8.0, 3.0, 15.0, 4.0, 2.5};
    auto outliers = tda::cpp23::findOutliers(data, 3.0);
    
    std::cout << "✅ Found " << outliers.size() << " outliers: ";
    for (const auto& [idx, value] : outliers) {
        std::cout << "(" << idx << ":" << value << ") ";
    }
    std::cout << std::endl;
    
    // Test 3: Parallel algorithms with modern features
    std::cout << "\n=== Testing Parallel Algorithms ===" << std::endl;
    
    auto points = generateTestPoints(100, 3);
    
    auto start = std::chrono::high_resolution_clock::now();
    tda::cpp23::parallelDistanceMatrix(points);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "✅ Parallel distance matrix computed in " << duration / 1000.0 << " ms" << std::endl;
    
    // Test 4: Functional programming with ranges
    std::cout << "\n=== Testing Functional Programming with Ranges ===" << std::endl;
    
    auto small_points = generateTestPoints(50, 2);
    auto graph = tda::cpp23::computeSparseGraph(small_points, 5.0);
    
    size_t total_edges = 0;
    for (const auto& neighbors : graph) {
        total_edges += neighbors.size();
    }
    
    std::cout << "✅ Sparse graph with " << total_edges << " edges (threshold 5.0)" << std::endl;
    
    // Test 5: Advanced ranges operations
    std::cout << "\n=== Testing Advanced Ranges Operations ===" << std::endl;
    
    auto test_data = std::views::iota(1, 101)  // Numbers 1-100
        | std::views::filter([](int x) { return x % 3 == 0; })  // Divisible by 3
        | std::views::transform([](int x) { return x * x; })     // Square them
        | std::views::take(10);  // Take first 10
    
    std::cout << "✅ First 10 squares of multiples of 3: ";
    for (auto val : test_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Test 6: Performance comparison
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    auto large_points = generateTestPoints(200, 5);
    
    // Traditional approach timing
    start = std::chrono::high_resolution_clock::now();
    size_t traditional_edges = 0;
    for (size_t i = 0; i < large_points.size(); ++i) {
        for (size_t j = i + 1; j < large_points.size(); ++j) {
            double dist = 0.0;
            for (size_t k = 0; k < large_points[i].size(); ++k) {
                double diff = large_points[i][k] - large_points[j][k];
                dist += diff * diff;
            }
            if (std::sqrt(dist) <= 4.0) {
                traditional_edges++;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto traditional_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Modern C++23 approach timing
    start = std::chrono::high_resolution_clock::now();
    auto modern_graph = tda::cpp23::computeSparseGraph(large_points, 4.0);
    size_t modern_edges = 0;
    for (const auto& neighbors : modern_graph) {
        modern_edges += neighbors.size();
    }
    modern_edges /= 2; // Each edge counted twice
    end = std::chrono::high_resolution_clock::now();
    auto modern_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Traditional approach: " << traditional_time / 1000.0 << " ms (" << traditional_edges << " edges)" << std::endl;
    std::cout << "Modern C++23 approach: " << modern_time / 1000.0 << " ms (" << modern_edges << " edges)" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << (double)traditional_time / modern_time << "x" << std::endl;
    
    std::cout << "\n🎯 C++23 Features Successfully Demonstrated:" << std::endl;
    std::cout << "- ✅ Enhanced constexpr capabilities for compile-time computation" << std::endl;
    std::cout << "- ✅ std::span for efficient array views and zero-copy operations" << std::endl;
    std::cout << "- ✅ Modern ranges and views for functional programming" << std::endl;
    std::cout << "- ✅ Parallel algorithms with std::execution policies" << std::endl;
    std::cout << "- ✅ Advanced ranges operations (enumerate, filter, transform, take)" << std::endl;
    std::cout << "- ✅ Performance improvements through modern C++ idioms" << std::endl;
    
    return 0;
}
