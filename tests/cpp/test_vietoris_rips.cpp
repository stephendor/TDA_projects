#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "tda/algorithms/vietoris_rips.hpp"
#include "tda/core/types.hpp"

namespace {
    void SetUp() override {
        // Create a simple 2D point cloud: square with diagonal
        // Points: (0,0), (1,0), (0,1), (1,1)
        // This will create interesting topology for testing
        points_ = {
            {0.0, 0.0},  // Point 0
            {1.0, 0.0},  // Point 1
            {0.0, 1.0},  // Point 2
            {1.0, 1.0}   // Point 3
        };
        
        // Threshold that will include the diagonal but not create 3-simplices
        threshold_ = 1.5;
        max_dimension_ = 2;
    }
    
    std::vector<std::vector<double>> points_;
    double threshold_;
    int max_dimension_;
};

TEST_F(VietorisRipsTest, Initialization) {
    tda::algorithms::VietorisRips vr;
    
    // Test successful initialization
    auto result = vr.initialize(points_, threshold_, max_dimension_);
    EXPECT_TRUE(result.has_value());
    
    // Test empty point cloud
    auto empty_result = vr.initialize({}, threshold_, max_dimension_);
    EXPECT_FALSE(empty_result.has_value());
    EXPECT_TRUE(empty_result.error().find("empty") != std::string::npos);
    
    // Test negative threshold
    auto neg_threshold_result = vr.initialize(points_, -1.0, max_dimension_);
    EXPECT_FALSE(neg_threshold_result.has_value());
    EXPECT_TRUE(neg_threshold_result.error().find("positive") != std::string::npos);
    
    // Test negative dimension
    auto neg_dim_result = vr.initialize(points_, threshold_, -1);
    EXPECT_FALSE(neg_dim_result.has_value());
    EXPECT_TRUE(neg_dim_result.error().find("non-negative") != std::string::npos);
    
    // Test invalid coefficient field
    auto invalid_field_result = vr.initialize(points_, threshold_, max_dimension_, 1);
    EXPECT_FALSE(invalid_field_result.has_value());
    EXPECT_TRUE(invalid_field_result.error().find("at least 2") != std::string::npos);
}

TEST_F(VietorisRipsTest, ComplexComputation) {
    tda::algorithms::VietorisRips vr;
    
    // Initialize
    auto init_result = vr.initialize(points_, threshold_, max_dimension_);
    ASSERT_TRUE(init_result.has_value());
    
    // Compute complex
    auto complex_result = vr.computeComplex();
    EXPECT_TRUE(complex_result.has_value());
    
    // Get statistics
    auto stats_result = vr.getStatistics();
    ASSERT_TRUE(stats_result.has_value());
    
    auto stats = stats_result.value();
    EXPECT_EQ(stats.num_points, 4);
    EXPECT_GT(stats.num_simplices, 0);
    EXPECT_EQ(stats.max_dimension, 2);
    EXPECT_DOUBLE_EQ(stats.threshold, threshold_);
    
    // Verify simplex counts by dimension
    EXPECT_EQ(stats.simplex_count_by_dim[0], 4);  // 4 vertices
    EXPECT_EQ(stats.simplex_count_by_dim[1], 6);  // 6 edges (including diagonal)
    EXPECT_EQ(stats.simplex_count_by_dim[2], 4);  // 4 triangles
}

TEST_F(VietorisRipsTest, PersistenceComputation) {
    tda::algorithms::VietorisRips vr;
    
    // Initialize and compute complex
    auto init_result = vr.initialize(points_, threshold_, max_dimension_);
    ASSERT_TRUE(init_result.has_value());
    
    auto complex_result = vr.computeComplex();
    ASSERT_TRUE(complex_result.has_value());
    
    // Compute persistence
    auto persistence_result = vr.computePersistence();
    EXPECT_TRUE(persistence_result.has_value());
    
    // Get persistence pairs
    auto pairs_result = vr.getPersistencePairs();
    ASSERT_TRUE(pairs_result.has_value());
    
    auto pairs = pairs_result.value();
    EXPECT_GT(pairs.size(), 0);
    
    // Verify persistence pair properties
    for (const auto& pair : pairs) {
        EXPECT_GE(pair.dimension, 0);
        EXPECT_LE(pair.dimension, max_dimension_);
        EXPECT_GE(pair.birth, 0.0);
        EXPECT_GT(pair.death, pair.birth);  // Death > birth
        EXPECT_DOUBLE_EQ(pair.persistence, pair.death - pair.birth);
    }
}

TEST_F(VietorisRipsTest, BettiNumbers) {
    tda::algorithms::VietorisRips vr;
    
    // Initialize, compute complex and persistence
    auto init_result = vr.initialize(points_, threshold_, max_dimension_);
    ASSERT_TRUE(init_result.has_value());
    
    auto complex_result = vr.computeComplex();
    ASSERT_TRUE(complex_result.has_value());
    
    auto persistence_result = vr.computePersistence();
    ASSERT_TRUE(persistence_result.has_value());
    
    // Get Betti numbers
    auto betti_result = vr.getBettiNumbers();
    ASSERT_TRUE(betti_result.has_value());
    
    auto betti_numbers = betti_result.value();
    EXPECT_EQ(betti_numbers.size(), max_dimension_ + 1);
    
    // β₀ should be 1 (one connected component)
    EXPECT_EQ(betti_numbers[0], 1);
    
    // β₁ should be at least 0 (no holes guaranteed)
    EXPECT_GE(betti_numbers[1], 0);
    
    // β₂ should be 0 (no 2D voids in 2D space)
    EXPECT_EQ(betti_numbers[2], 0);
}

TEST_F(VietorisRipsTest, SimplexRetrieval) {
    tda::algorithms::VietorisRips vr;
    
    // Initialize and compute complex
    auto init_result = vr.initialize(points_, threshold_, max_dimension_);
    ASSERT_TRUE(init_result.has_value());
    
    auto complex_result = vr.computeComplex();
    ASSERT_TRUE(complex_result.has_value());
    
    // Get simplices
    auto simplices_result = vr.getSimplices();
    ASSERT_TRUE(simplices_result.has_value());
    
    auto simplices = simplices_result.value();
    EXPECT_GT(simplices.size(), 0);
    
    // Verify simplex properties
    for (const auto& simplex : simplices) {
        EXPECT_GE(simplex.dimension, 0);
        EXPECT_LE(simplex.dimension, max_dimension_);
        EXPECT_GE(simplex.filtration_value, 0.0);
        EXPECT_EQ(simplex.vertices.size(), simplex.dimension + 1);
        
        // Verify vertex indices are valid
        for (int vertex : simplex.vertices) {
            EXPECT_GE(vertex, 0);
            EXPECT_LT(vertex, static_cast<int>(points_.size()));
        }
    }
}

TEST_F(VietorisRipsTest, DistanceComputation) {
    tda::algorithms::VietorisRips vr;
    
    // Test batch distance computation
    std::vector<double> query_point = {0.5, 0.5};  // Center of the square
    
    auto distances = vr.computeDistancesBatch(points_, query_point);
    EXPECT_EQ(distances.size(), points_.size());
    
    // Verify distances are reasonable
    for (size_t i = 0; i < distances.size(); ++i) {
        EXPECT_GE(distances[i], 0.0);
        
        // Manual distance calculation for verification
        double expected_dist = 0.0;
        for (size_t j = 0; j < points_[i].size(); ++j) {
            double diff = points_[i][j] - query_point[j];
            expected_dist += diff * diff;
        }
        expected_dist = std::sqrt(expected_dist);
        
        EXPECT_NEAR(distances[i], expected_dist, 1e-10);
    }
}

TEST_F(VietorisRipsTest, ErrorHandling) {
    tda::algorithms::VietorisRips vr;
    
    // Try to get simplices before computing complex
    auto simplices_result = vr.getSimplices();
    EXPECT_FALSE(simplices_result.has_value());
    EXPECT_TRUE(simplices_result.error().find("not computed") != std::string::npos);
    
    // Try to get persistence pairs before computing persistence
    auto init_result = vr.initialize(points_, threshold_, max_dimension_);
    ASSERT_TRUE(init_result.has_value());
    
    auto complex_result = vr.computeComplex();
    ASSERT_TRUE(complex_result.has_value());
    
    auto pairs_result = vr.getPersistencePairs();
    EXPECT_FALSE(pairs_result.has_value());
    EXPECT_TRUE(pairs_result.error().find("not computed") != std::string::npos);
    
    // Try to get Betti numbers before computing persistence
    auto betti_result = vr.getBettiNumbers();
    EXPECT_FALSE(betti_result.has_value());
    EXPECT_TRUE(betti_result.error().find("not computed") != std::string::npos);
}

TEST_F(VietorisRipsTest, MoveSemantics) {
    tda::algorithms::VietorisRips vr1;
    
    // Initialize and compute complex
    auto init_result = vr1.initialize(points_, threshold_, max_dimension_);
    ASSERT_TRUE(init_result.has_value());
    
    auto complex_result = vr1.computeComplex();
    ASSERT_TRUE(complex_result.has_value());
    
    // Move to new instance
    tda::algorithms::VietorisRips vr2 = std::move(vr1);
    
    // Original should be in moved-from state
    auto stats_result1 = vr1.getStatistics();
    EXPECT_FALSE(stats_result1.has_value());
    
    // New instance should work
    auto stats_result2 = vr2.getStatistics();
    EXPECT_TRUE(stats_result2.has_value());
    
    auto stats = stats_result2.value();
    EXPECT_EQ(stats.num_points, 4);
    EXPECT_GT(stats.num_simplices, 0);
}

TEST_F(VietorisRipsTest, LargePointCloud) {
    // Create a larger point cloud for performance testing
    std::vector<std::vector<double>> large_points;
    const int num_points = 100;
    const int dimension = 3;
    
    for (int i = 0; i < num_points; ++i) {
        std::vector<double> point;
        for (int j = 0; j < dimension; ++j) {
            point.push_back(static_cast<double>(i + j) / num_points);
        }
        large_points.push_back(std::move(point));
    }
    
    tda::algorithms::VietorisRips vr;
    
    // Test with larger point cloud
    auto init_result = vr.initialize(large_points, 0.5, 2);
    ASSERT_TRUE(init_result.has_value());
    
    auto complex_result = vr.computeComplex();
    ASSERT_TRUE(complex_result.has_value());
    
    auto stats_result = vr.getStatistics();
    ASSERT_TRUE(stats_result.has_value());
    
    auto stats = stats_result.value();
    EXPECT_EQ(stats.num_points, num_points);
    EXPECT_GT(stats.num_simplices, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
