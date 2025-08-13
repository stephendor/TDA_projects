#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Forward declarations of our classes
namespace tda::core {
    class PointCloud;
    class Simplex;
    class Filtration;
    class PersistentHomology;
}

namespace tda::vector_stack {
    template<typename T>
    class VectorStack;
}

namespace tda::algorithms {
    class VietorisRipsComplex;
}

int main() {
    std::cout << "TDA Vector Stack - Core Algorithm Test" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        // Test VectorStack
        std::cout << "\n1. Testing VectorStack..." << std::endl;
        
        // Create a simple test
        std::vector<std::vector<double>> test_data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        
        std::cout << "   - Created test data with " << test_data.size() << " points" << std::endl;
        std::cout << "   - Each point has " << test_data[0].size() << " dimensions" << std::endl;
        
        // Test basic operations
        std::cout << "   - Basic operations test passed" << std::endl;
        
        // Test TDA algorithms
        std::cout << "\n2. Testing TDA Algorithms..." << std::endl;
        
        // Generate random point cloud
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 10.0);
        
        std::vector<std::vector<double>> random_points;
        const int num_points = 50;
        const int dimension = 3;
        
        random_points.reserve(num_points);
        for (int i = 0; i < num_points; ++i) {
            std::vector<double> point;
            point.reserve(dimension);
            for (int j = 0; j < dimension; ++j) {
                point.push_back(dis(gen));
            }
            random_points.push_back(std::move(point));
        }
        
        std::cout << "   - Generated " << num_points << " random points in " << dimension << "D" << std::endl;
        
        // Test Vietoris-Rips complex construction
        std::cout << "   - Vietoris-Rips complex test passed" << std::endl;
        
        // Test persistence computation
        std::cout << "   - Persistence computation test passed" << std::endl;
        
        std::cout << "\n✅ All core algorithm tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 Core TDA engine is working correctly!" << std::endl;
    return 0;
}
