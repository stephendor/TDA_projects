#include <iostream>
#include <vector>
#include <cmath>
#include "tda/algorithms/vietoris_rips.hpp"

int main() {
    std::cout << "🔬 Topology Debugging - Step by Step Analysis" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Create simple circle points
    std::vector<std::vector<double>> points;
    int n_points = 6; // Small number for debugging
    for (int i = 0; i < n_points; ++i) {
        double angle = 2.0 * M_PI * i / n_points;
        points.push_back({std::cos(angle), std::sin(angle)});
        std::cout << "Point " << i << ": (" << points[i][0] << ", " << points[i][1] << ")" << std::endl;
    }
    
    // Calculate distances to understand appropriate threshold
    std::cout << "\n📏 Distance Matrix:" << std::endl;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dx = points[i][0] - points[j][0];
            double dy = points[i][1] - points[j][1];
            double dist = std::sqrt(dx*dx + dy*dy);
            std::cout << "d(" << i << "," << j << ") = " << dist << std::endl;
        }
    }
    
    // Test different thresholds
    std::vector<double> thresholds = {0.5, 1.0, 1.2, 1.5, 2.0};
    
    for (double threshold : thresholds) {
        std::cout << "\n🧪 Testing threshold = " << threshold << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        tda::algorithms::VietorisRips vr;
        auto init_result = vr.initialize(points, threshold, 2, 2);
        
        if (!init_result.has_value()) {
            std::cout << "❌ Initialization failed: " << init_result.error() << std::endl;
            continue;
        }
        
        auto complex_result = vr.computeComplex();
        if (!complex_result.has_value()) {
            std::cout << "❌ Complex computation failed: " << complex_result.error() << std::endl;
            continue;
        }
        
        // Get complex statistics before persistence
        auto stats_result = vr.getStatistics();
        if (stats_result.has_value()) {
            const auto& stats = stats_result.value();
            std::cout << "Complex built successfully:" << std::endl;
            std::cout << "  Points: " << stats.num_points << std::endl;
            std::cout << "  Total simplices: " << stats.num_simplices << std::endl;
            std::cout << "  Max dimension: " << stats.max_dimension << std::endl;
            
            if (stats.simplex_count_by_dim.size() > 0) {
                std::cout << "  Simplex counts by dimension: ";
                for (size_t i = 0; i < stats.simplex_count_by_dim.size(); ++i) {
                    std::cout << "dim" << i << "=" << stats.simplex_count_by_dim[i] << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // Now compute persistence
        auto persistence_result = vr.computePersistence();
        if (!persistence_result.has_value()) {
            std::cout << "❌ Persistence computation failed: " << persistence_result.error() << std::endl;
            continue;
        }
        
        // Get persistence pairs to debug
        auto pairs_result = vr.getPersistencePairs();
        if (pairs_result.has_value()) {
            const auto& pairs = pairs_result.value();
            std::cout << "Persistence pairs (" << pairs.size() << " total):" << std::endl;
            for (size_t i = 0; i < pairs.size(); ++i) {
                const auto& pair = pairs[i];
                std::cout << "  Pair " << i << ": dim=" << pair.dimension 
                          << " birth=" << pair.birth << " death=" << pair.death 
                          << " persistence=" << pair.get_persistence() << std::endl;
            }
        }
        
        // Get Betti numbers
        auto betti_result = vr.getBettiNumbers();
        if (betti_result.has_value()) {
            const auto& betti_numbers = betti_result.value();
            std::cout << "Betti numbers: ";
            for (size_t i = 0; i < betti_numbers.size(); ++i) {
                std::cout << "H" << i << "=" << betti_numbers[i] << " ";
            }
            std::cout << std::endl;
            
            // Expected: H0=1 (connected), H1=1 (one hole) for circle
            if (betti_numbers.size() > 0 && betti_numbers[0] == 1) {
                std::cout << "✅ H0=1 (connected) ✓" << std::endl;
            } else if (betti_numbers.size() > 0) {
                std::cout << "❌ H0=" << betti_numbers[0] << " (expected 1)" << std::endl;
            }
            
            if (betti_numbers.size() > 1 && betti_numbers[1] == 1) {
                std::cout << "✅ H1=1 (one hole) ✓" << std::endl;
            } else if (betti_numbers.size() > 1) {
                std::cout << "⚠️  H1=" << betti_numbers[1] << " (expected 1)" << std::endl;
            }
            
        } else {
            std::cout << "❌ Betti number computation failed: " << betti_result.error() << std::endl;
        }
    }
    
    return 0;
}

