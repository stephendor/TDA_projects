#include "tda/algorithms/vietoris_rips.hpp"
#include "tda/core/types.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printPointCloud(const std::vector<std::vector<double>>& points) {
    std::cout << "Point Cloud (" << points.size() << " points):" << std::endl;
    for (size_t i = 0; i < points.size(); ++i) {
        std::cout << "  Point " << i << ": (";
        for (size_t j = 0; j < points[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << points[i][j];
        }
        std::cout << ")" << std::endl;
    }
}

void printStatistics(const tda::ComplexStatistics& stats) {
    std::cout << "\nComplex Statistics:" << std::endl;
    std::cout << "  Number of points: " << stats.num_points << std::endl;
    std::cout << "  Number of simplices: " << stats.num_simplices << std::endl;
    std::cout << "  Maximum dimension: " << stats.max_dimension << std::endl;
    std::cout << "  Threshold: " << std::fixed << std::setprecision(3) << stats.threshold << std::endl;
    
    std::cout << "  Simplex count by dimension:" << std::endl;
    for (size_t i = 0; i < stats.simplex_count_by_dim.size(); ++i) {
        std::cout << "    Dimension " << i << ": " << stats.simplex_count_by_dim[i] << " simplices" << std::endl;
    }
}

void printSimplices(const std::vector<tda::SimplexInfo>& simplices) {
    std::cout << "\nSimplices:" << std::endl;
    for (size_t i = 0; i < simplices.size(); ++i) {
        const auto& simplex = simplices[i];
        std::cout << "  Simplex " << i << " (dim " << simplex.dimension << "): ";
        std::cout << "filtration=" << std::fixed << std::setprecision(3) << simplex.filtration_value;
        std::cout << ", vertices=[";
        for (size_t j = 0; j < simplex.vertices.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << simplex.vertices[j];
        }
        std::cout << "]" << std::endl;
    }
}

void printPersistencePairs(const std::vector<tda::PersistencePair>& pairs) {
    std::cout << "\nPersistence Pairs:" << std::endl;
    std::cout << "  Dimension | Birth    | Death    | Persistence" << std::endl;
    std::cout << "  ----------|----------|----------|------------" << std::endl;
    
    for (const auto& pair : pairs) {
        std::cout << "  " << std::setw(9) << pair.dimension << " | ";
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << pair.birth << " | ";
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << pair.death << " | ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(3) << pair.persistence << std::endl;
    }
}

void printBettiNumbers(const std::vector<int>& betti_numbers) {
    std::cout << "\nBetti Numbers:" << std::endl;
    for (size_t i = 0; i < betti_numbers.size(); ++i) {
        std::cout << "  β" << i << " = " << betti_numbers[i];
        if (i == 0) std::cout << " (connected components)";
        else if (i == 1) std::cout << " (holes/loops)";
        else if (i == 2) std::cout << " (voids/cavities)";
        std::cout << std::endl;
    }
}

int main() {
    printHeader("TDA Vector Stack - Vietoris-Rips Demo");
    
    // Create a simple 2D point cloud: square with diagonal
    // This will create interesting topology for demonstration
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},  // Point 0: bottom-left
        {1.0, 0.0},  // Point 1: bottom-right
        {0.0, 1.0},  // Point 2: top-left
        {1.0, 1.0}   // Point 3: top-right
    };
    
    printPointCloud(points);
    
    // Set parameters
    double threshold = 1.5;  // Will include diagonal but not create 3-simplices
    int max_dimension = 2;
    int coefficient_field = 2;  // Z/2Z coefficients
    
    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Threshold: " << threshold << std::endl;
    std::cout << "  Max dimension: " << max_dimension << std::endl;
    std::cout << "  Coefficient field: Z/" << coefficient_field << "Z" << std::endl;
    
    // Create and initialize Vietoris-Rips algorithm
    printHeader("Initializing Vietoris-Rips Algorithm");
    
    tda::algorithms::VietorisRips vr;
    
    auto init_result = vr.initialize(points, threshold, max_dimension, coefficient_field);
    if (!init_result.has_value()) {
        std::cerr << "❌ Initialization failed: " << init_result.error() << std::endl;
        return 1;
    }
    std::cout << "✅ Initialization successful" << std::endl;
    
    // Compute the complex
    printHeader("Computing Vietoris-Rips Complex");
    
    auto complex_result = vr.computeComplex();
    if (!complex_result.has_value()) {
        std::cerr << "❌ Complex computation failed: " << complex_result.error() << std::endl;
        return 1;
    }
    std::cout << "✅ Complex computation successful" << std::endl;
    
    // Get and display statistics
    auto stats_result = vr.getStatistics();
    if (!stats_result.has_value()) {
        std::cerr << "❌ Failed to get statistics: " << stats_result.error() << std::endl;
        return 1;
    }
    
    printStatistics(stats_result.value());
    
    // Get and display simplices
    auto simplices_result = vr.getSimplices();
    if (!simplices_result.has_value()) {
        std::cerr << "❌ Failed to get simplices: " << simplices_result.error() << std::endl;
        return 1;
    }
    
    printSimplices(simplices_result.value());
    
    // Compute persistent homology
    printHeader("Computing Persistent Homology");
    
    auto persistence_result = vr.computePersistence();
    if (!persistence_result.has_value()) {
        std::cerr << "❌ Persistence computation failed: " << persistence_result.error() << std::endl;
        return 1;
    }
    std::cout << "✅ Persistence computation successful" << std::endl;
    
    // Get and display persistence pairs
    auto pairs_result = vr.getPersistencePairs();
    if (!pairs_result.has_value()) {
        std::cerr << "❌ Failed to get persistence pairs: " << pairs_result.error() << std::endl;
        return 1;
    }
    
    printPersistencePairs(pairs_result.value());
    
    // Get and display Betti numbers
    auto betti_result = vr.getBettiNumbers();
    if (!betti_result.has_value()) {
        std::cerr << "❌ Failed to get Betti numbers: " << betti_result.error() << std::endl;
        return 1;
    }
    
    printBettiNumbers(betti_result.value());
    
    // Test distance computation
    printHeader("Testing Distance Computation");
    
    std::vector<double> query_point = {0.5, 0.5};  // Center of the square
    std::cout << "Query point: (";
    for (size_t i = 0; i < query_point.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(2) << query_point[i];
    }
    std::cout << ")" << std::endl;
    
    auto distances = vr.computeDistancesBatch(points, query_point);
    std::cout << "Distances to query point:" << std::endl;
    for (size_t i = 0; i < distances.size(); ++i) {
        std::cout << "  Point " << i << ": " << std::fixed << std::setprecision(3) << distances[i] << std::endl;
    }
    
    // Summary
    printHeader("Demo Summary");
    std::cout << "✅ Successfully demonstrated Vietoris-Rips algorithm with:" << std::endl;
    std::cout << "  - Complex construction from point cloud" << std::endl;
    std::cout << "  - Persistent homology computation" << std::endl;
    std::endl;
    std::cout << "  - Betti number calculation" << std::endl;
    std::cout << "  - SIMD-optimized distance computations" << std::endl;
    std::cout << "  - Comprehensive error handling" << std::endl;
    std::cout << "  - Modern C++23 features and GUDHI integration" << std::endl;
    
    std::cout << "\n🚀 Ready for advanced TDA development!" << std::endl;
    
    return 0;
}
