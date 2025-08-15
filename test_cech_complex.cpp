#include "tda/algorithms/cech_complex.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace tda::algorithms;

void testBasicInitialization() {
    std::cout << "Testing basic initialization..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    
    auto result = cech.initialize(points);
    assert(result.has_value());
    std::cout << "✓ Basic initialization successful" << std::endl;
}

void testEmptyPointCloud() {
    std::cout << "Testing empty point cloud handling..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> emptyPoints;
    
    auto result = cech.initialize(emptyPoints);
    assert(result.has_error());
    std::cout << "✓ Empty point cloud properly rejected" << std::endl;
}

void testZeroDimensionPoints() {
    std::cout << "Testing zero dimension points handling..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> invalidPoints = {{}};
    
    auto result = cech.initialize(invalidPoints);
    assert(result.has_error());
    std::cout << "✓ Zero dimension points properly rejected" << std::endl;
}

void testComplexComputation() {
    std::cout << "Testing complex computation..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    std::cout << "✓ Complex computation successful" << std::endl;
}

void testPersistenceComputation() {
    std::cout << "Testing persistence computation..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    auto persistenceResult = cech.computePersistence();
    assert(persistenceResult.has_value());
    std::cout << "✓ Persistence computation successful" << std::endl;
}

void testSimplicesRetrieval() {
    std::cout << "Testing simplices retrieval..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    auto simplicesResult = cech.getSimplices();
    assert(simplicesResult.has_value());
    
    auto simplices = simplicesResult.value();
    assert(!simplices.empty());
    
    // Should have at least vertices (0-simplices)
    bool hasVertices = false;
    for (const auto& simplex : simplices) {
        if (simplex.dimension == 0) {
            hasVertices = true;
            break;
        }
    }
    assert(hasVertices);
    
    std::cout << "✓ Simplices retrieval successful, found " << simplices.size() << " simplices" << std::endl;
}

void testPersistencePairsRetrieval() {
    std::cout << "Testing persistence pairs retrieval..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    auto persistenceResult = cech.computePersistence();
    assert(persistenceResult.has_value());
    
    auto pairsResult = cech.getPersistencePairs();
    assert(pairsResult.has_value());
    
    auto pairs = pairsResult.value();
    std::cout << "✓ Persistence pairs retrieval successful, found " << pairs.size() << " pairs" << std::endl;
}

void testBettiNumbers() {
    std::cout << "Testing Betti numbers computation..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    auto persistenceResult = cech.computePersistence();
    assert(persistenceResult.has_value());
    
    auto bettiResult = cech.getBettiNumbers();
    assert(bettiResult.has_value());
    
    auto bettiNumbers = bettiResult.value();
    assert(!bettiNumbers.empty());
    
    std::cout << "✓ Betti numbers computation successful: ";
    for (size_t i = 0; i < bettiNumbers.size(); ++i) {
        std::cout << "β" << i << "=" << bettiNumbers[i];
        if (i < bettiNumbers.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

void testStatistics() {
    std::cout << "Testing statistics computation..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    auto statsResult = cech.getStatistics();
    assert(statsResult.has_value());
    
    auto stats = statsResult.value();
    assert(stats.num_points == points.size());
    assert(stats.num_simplices > 0);
    
    std::cout << "✓ Statistics computation successful:" << std::endl;
    std::cout << "  - Points: " << stats.num_points << std::endl;
    std::cout << "  - Simplices: " << stats.num_simplices << std::endl;
    std::cout << "  - Max dimension: " << stats.max_dimension << std::endl;
    std::cout << "  - Simplex count by dimension: ";
    for (size_t i = 0; i < stats.simplex_count_by_dim.size(); ++i) {
        std::cout << "dim" << i << "=" << stats.simplex_count_by_dim[i];
        if (i < stats.simplex_count_by_dim.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

void testConfiguration() {
    std::cout << "Testing configuration management..." << std::endl;
    
    CechComplex::Config config;
    config.radius = 2.0;
    config.maxDimension = 2;
    config.maxNeighbors = 25;
    
    CechComplex cech(config);
    auto currentConfig = cech.getConfig();
    
    assert(currentConfig.radius == 2.0);
    assert(currentConfig.maxDimension == 2);
    assert(currentConfig.maxNeighbors == 25);
    
    // Test configuration update
    CechComplex::Config newConfig;
    newConfig.radius = 3.0;
    newConfig.maxDimension = 4;
    
    cech.updateConfig(newConfig);
    auto updatedConfig = cech.getConfig();
    
    assert(updatedConfig.radius == 3.0);
    assert(updatedConfig.maxDimension == 4);
    
    std::cout << "✓ Configuration management successful" << std::endl;
}

void testClearFunctionality() {
    std::cout << "Testing clear functionality..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    
    auto initResult = cech.initialize(points);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    cech.clear();
    
    // After clear, getSimplices should fail
    auto simplicesResult = cech.getSimplices();
    assert(simplicesResult.has_error());
    
    std::cout << "✓ Clear functionality successful" << std::endl;
}

void testMoveSemantics() {
    std::cout << "Testing move semantics..." << std::endl;
    
    CechComplex cech1;
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    
    auto initResult = cech1.initialize(points);
    assert(initResult.has_value());
    
    // Test move constructor
    CechComplex cech2(std::move(cech1));
    
    // cech1 should be in moved-from state
    auto simplicesResult1 = cech1.getSimplices();
    assert(simplicesResult1.has_error());
    
    // cech2 should work
    auto computeResult = cech2.computeComplex();
    assert(computeResult.has_value());
    
    std::cout << "✓ Move semantics successful" << std::endl;
}

void test3DPointCloud() {
    std::cout << "Testing 3D point cloud..." << std::endl;
    
    CechComplex cech;
    std::vector<std::vector<double>> points3D = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}
    };
    
    auto initResult = cech.initialize(points3D);
    assert(initResult.has_value());
    
    auto computeResult = cech.computeComplex();
    assert(computeResult.has_value());
    
    auto persistenceResult = cech.computePersistence();
    assert(persistenceResult.has_value());
    
    auto statsResult = cech.getStatistics();
    assert(statsResult.has_value());
    
    auto stats = statsResult.value();
    std::cout << "✓ 3D point cloud successful:" << std::endl;
    std::cout << "  - Points: " << stats.num_points << std::endl;
    std::cout << "  - Simplices: " << stats.num_simplices << std::endl;
    std::cout << "  - Max dimension: " << stats.max_dimension << std::endl;
}

int main() {
    std::cout << "Starting Čech Complex tests..." << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        testBasicInitialization();
        testEmptyPointCloud();
        testZeroDimensionPoints();
        testComplexComputation();
        testPersistenceComputation();
        testSimplicesRetrieval();
        testPersistencePairsRetrieval();
        testBettiNumbers();
        testStatistics();
        testConfiguration();
        testClearFunctionality();
        testMoveSemantics();
        test3DPointCloud();
        
        std::cout << "=================================" << std::endl;
        std::cout << "All tests passed successfully! 🎉" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
