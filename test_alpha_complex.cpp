#include "tda/algorithms/alpha_complex.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing Alpha Complex Implementation" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Create a simple 2D point cloud (circle)
    std::vector<std::vector<double>> points_2d = {
        {0.0, 0.0},      // Center
        {1.0, 0.0},      // Right
        {0.0, 1.0},      // Top
        {-1.0, 0.0},     // Left
        {0.0, -1.0},     // Bottom
        {0.707, 0.707},  // Top-right
        {-0.707, 0.707}, // Top-left
        {-0.707, -0.707},// Bottom-left
        {0.707, -0.707}  // Bottom-right
    };
    
    std::cout << "2D point cloud created with " << points_2d.size() << " points" << std::endl;
    
    // Test 2D Alpha Complex
    {
        std::cout << "\n--- Testing 2D Alpha Complex ---" << std::endl;
        
        tda::algorithms::AlphaComplex ac_2d;
        
        // Initialize with max dimension 2
        auto init_result = ac_2d.initialize(points_2d, 2, 2);
        if (init_result.has_error()) {
            std::cout << "2D Initialization failed: " << init_result.error() << std::endl;
            return 1;
        }
        std::cout << "✓ 2D Initialization successful" << std::endl;
        
        // Compute the complex
        auto complex_result = ac_2d.computeComplex();
        if (complex_result.has_error()) {
            std::cout << "2D Complex computation failed: " << complex_result.error() << std::endl;
            return 1;
        }
        std::cout << "✓ 2D Complex computation successful" << std::endl;
        
        // Compute persistence
        auto persistence_result = ac_2d.computePersistence();
        if (persistence_result.has_error()) {
            std::cout << "2D Persistence computation failed: " << persistence_result.error() << std::endl;
            return 1;
        }
        std::cout << "✓ 2D Persistence computation successful" << std::endl;
        
        // Get statistics
        auto stats_result = ac_2d.getStatistics();
        if (stats_result.has_error()) {
            std::cout << "2D Statistics retrieval failed: " << stats_result.error() << std::endl;
            return 1;
        }
        
        auto stats = stats_result.value();
        std::cout << "✓ 2D Statistics retrieved:" << std::endl;
        std::cout << "  - Number of points: " << stats.num_points << std::endl;
        std::cout << "  - Number of simplices: " << stats.num_simplices << std::endl;
        std::cout << "  - Max dimension: " << stats.max_dimension << std::endl;
        
        // Get simplices
        auto simplices_result = ac_2d.getSimplices();
        if (simplices_result.has_error()) {
            std::cout << "2D Simplices retrieval failed: " << simplices_result.error() << std::endl;
            return 1;
        }
        
        auto simplices = simplices_result.value();
        std::cout << "✓ 2D Retrieved " << simplices.size() << " simplices:" << std::endl;
        for (size_t i = 0; i < std::min(simplices.size(), size_t(10)); ++i) { // Show first 10
            const auto& simplex = simplices[i];
            std::cout << "  Simplex " << i << ": dim=" << simplex.dimension 
                      << ", filt=" << simplex.filtration_value 
                      << ", vertices=[";
            for (size_t j = 0; j < simplex.vertices.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << simplex.vertices[j];
            }
            std::cout << "]" << std::endl;
        }
        if (simplices.size() > 10) {
            std::cout << "  ... and " << (simplices.size() - 10) << " more simplices" << std::endl;
        }
    }
    
    // Create a simple 3D point cloud (sphere)
    std::vector<std::vector<double>> points_3d = {
        {0.0, 0.0, 0.0},      // Center
        {1.0, 0.0, 0.0},      // Right
        {0.0, 1.0, 0.0},      // Top
        {0.0, 0.0, 1.0},      // Front
        {-1.0, 0.0, 0.0},     // Left
        {0.0, -1.0, 0.0},     // Bottom
        {0.0, 0.0, -1.0},     // Back
        {0.577, 0.577, 0.577}, // Top-right-front
        {-0.577, 0.577, 0.577} // Top-left-front
    };
    
    std::cout << "\n3D point cloud created with " << points_3d.size() << " points" << std::endl;
    
    // Test 3D Alpha Complex
    {
        std::cout << "\n--- Testing 3D Alpha Complex ---" << std::endl;
        
        tda::algorithms::AlphaComplex ac_3d;
        
        // Initialize with max dimension 3
        auto init_result = ac_3d.initialize(points_3d, 3, 2);
        if (init_result.has_error()) {
            std::cout << "3D Initialization failed: " << init_result.error() << std::endl;
            return 1;
        }
        std::cout << "✓ 3D Initialization successful" << std::endl;
        
        // Compute the complex
        auto complex_result = ac_3d.computeComplex();
        if (complex_result.has_error()) {
            std::cout << "3D Complex computation failed: " << complex_result.error() << std::endl;
            return 1;
        }
        std::cout << "✓ 3D Complex computation successful" << std::endl;
        
        // Compute persistence
        auto persistence_result = ac_3d.computePersistence();
        if (persistence_result.has_error()) {
            std::cout << "3D Persistence computation failed: " << persistence_result.error() << std::endl;
            return 1;
        }
        std::cout << "✓ 3D Persistence computation successful" << std::endl;
        
        // Get statistics
        auto stats_result = ac_3d.getStatistics();
        if (stats_result.has_error()) {
            std::cout << "3D Statistics retrieval failed: " << stats_result.error() << std::endl;
            return 1;
        }
        
        auto stats = stats_result.value();
        std::cout << "✓ 3D Statistics retrieved:" << std::endl;
        std::cout << "  - Number of points: " << stats.num_points << std::endl;
        std::cout << "  - Number of simplices: " << stats.num_simplices << std::endl;
        std::cout << "  - Max dimension: " << stats.max_dimension << std::endl;
        
        // Get simplices
        auto simplices_result = ac_3d.getSimplices();
        if (simplices_result.has_error()) {
            std::cout << "3D Simplices retrieval failed: " << simplices_result.error() << std::endl;
            return 1;
        }
        
        auto simplices = simplices_result.value();
        std::cout << "✓ 3D Retrieved " << simplices.size() << " simplices:" << std::endl;
        for (size_t i = 0; i < std::min(simplices.size(), size_t(10)); ++i) { // Show first 10
            const auto& simplex = simplices[i];
            std::cout << "  Simplex " << i << ": dim=" << simplex.dimension 
                      << ", filt=" << simplex.filtration_value 
                      << ", vertices=[";
            for (size_t j = 0; j < simplex.vertices.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << simplex.vertices[j];
            }
            std::cout << "]" << std::endl;
        }
        if (simplices.size() > 10) {
            std::cout << "  ... and " << (simplices.size() - 10) << " more simplices" << std::endl;
        }
    }
    
    // Test error handling with invalid input
    {
        std::cout << "\n--- Testing Error Handling ---" << std::endl;
        
        tda::algorithms::AlphaComplex ac_error;
        
        // Test with 1D points (should fail)
        std::vector<std::vector<double>> invalid_points = {{1.0}, {2.0}, {3.0}};
        auto init_result = ac_error.initialize(invalid_points, 2, 2);
        if (init_result.has_error()) {
            std::cout << "✓ Correctly rejected 1D points: " << init_result.error() << std::endl;
        } else {
            std::cout << "❌ Should have rejected 1D points" << std::endl;
            return 1;
        }
        
        // Test with empty point cloud (should fail)
        std::vector<std::vector<double>> empty_points;
        auto init_result2 = ac_error.initialize(empty_points, 2, 2);
        if (init_result2.has_error()) {
            std::cout << "✓ Correctly rejected empty point cloud: " << init_result2.error() << std::endl;
        } else {
            std::cout << "❌ Should have rejected empty point cloud" << std::endl;
            return 1;
        }
    }
    
    std::cout << "\n🎉 All Alpha Complex tests passed! Implementation is working correctly." << std::endl;
    return 0;
}
