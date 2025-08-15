#include <iostream>
#include <vector>
#include <cmath>

// Direct GUDHI includes to test integration
#include <gudhi/Simplex_tree.h>
#include <gudhi/Rips_complex.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/distance_functions.h>

int main() {
    std::cout << "🔬 Direct GUDHI Integration Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Create simple triangle (3 points)
    std::vector<std::vector<double>> points = {
        {0.0, 0.0},
        {1.0, 0.0}, 
        {0.5, 0.866}
    };
    
    std::cout << "Points:" << std::endl;
    for (size_t i = 0; i < points.size(); ++i) {
        std::cout << "  " << i << ": (" << points[i][0] << ", " << points[i][1] << ")" << std::endl;
    }
    
    double threshold = 1.5; // Should connect all points
    
    try {
        // Step 1: Create Rips complex
        std::cout << "\n📐 Step 1: Creating Rips complex..." << std::endl;
        using Rips_complex = Gudhi::rips_complex::Rips_complex<double>;
        Rips_complex rips_complex(points, threshold, Gudhi::Euclidean_distance());
        std::cout << "✅ Rips complex created" << std::endl;
        
        // Step 2: Create simplex tree
        std::cout << "\n🌳 Step 2: Creating simplex tree..." << std::endl;
        using Simplex_tree = Gudhi::Simplex_tree<>;
        Simplex_tree simplex_tree;
        
        // Step 3: Build complex
        std::cout << "\n🔨 Step 3: Building complex (max_dim=2)..." << std::endl;
        rips_complex.create_complex(simplex_tree, 2);
        
        std::cout << "Complex statistics:" << std::endl;
        std::cout << "  Num simplices: " << simplex_tree.num_simplices() << std::endl;
        std::cout << "  Max dimension: " << simplex_tree.dimension() << std::endl;
        
        // Count simplices by dimension
        std::vector<int> simplex_counts(3, 0);
        for (auto simplex : simplex_tree.complex_simplex_range()) {
            int dim = simplex_tree.dimension(simplex);
            if (dim >= 0 && dim < 3) simplex_counts[dim]++;
        }
        std::cout << "  By dimension: ";
        for (int i = 0; i < 3; ++i) {
            std::cout << "dim" << i << "=" << simplex_counts[i] << " ";
        }
        std::cout << std::endl;
        
        // Step 4: Check filtration values
        std::cout << "\n🔍 Step 4: Checking filtration values..." << std::endl;
        bool has_filtration = false;
        for (auto simplex : simplex_tree.complex_simplex_range()) {
            double filt_val = simplex_tree.filtration(simplex);
            int dim = simplex_tree.dimension(simplex);
            std::cout << "  Simplex dim=" << dim << " filtration=" << filt_val << std::endl;
            if (filt_val > 0) has_filtration = true;
        }
        
        if (!has_filtration) {
            std::cout << "❌ WARNING: No positive filtration values found!" << std::endl;
        }
        
        // Step 5: Create persistent cohomology
        std::cout << "\n⚗️  Step 5: Creating persistent cohomology..." << std::endl;
        using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
        using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;
        
        Persistent_cohomology persistent_cohomology(simplex_tree);
        std::cout << "✅ Persistent cohomology object created" << std::endl;
        
        // Step 6: Initialize coefficients
        std::cout << "\n🔢 Step 6: Initializing coefficients..." << std::endl;
        persistent_cohomology.init_coefficients(2); // Z/2Z field
        std::cout << "✅ Coefficients initialized" << std::endl;
        
        // Step 7: Compute persistence
        std::cout << "\n⚡ Step 7: Computing persistent cohomology..." << std::endl;
        persistent_cohomology.compute_persistent_cohomology();
        std::cout << "✅ Persistence computation completed" << std::endl;
        
        // Step 8: Get persistence pairs
        std::cout << "\n📊 Step 8: Getting persistence pairs..." << std::endl;
        auto persistent_pairs = persistent_cohomology.get_persistent_pairs();
        std::cout << "Number of persistence pairs: " << persistent_pairs.size() << std::endl;
        
        for (size_t i = 0; i < persistent_pairs.size(); ++i) {
            auto pair = persistent_pairs[i];
            auto birth_handle = std::get<0>(pair);
            auto death_handle = std::get<1>(pair);
            double birth = simplex_tree.filtration(birth_handle);
            double death = simplex_tree.filtration(death_handle);
            int dim = simplex_tree.dimension(birth_handle);
            
            std::cout << "  Pair " << i << ": dim=" << dim 
                      << " birth=" << birth << " death=" << death 
                      << " persistence=" << (death - birth) << std::endl;
        }
        
        // Step 9: Get Betti numbers
        std::cout << "\n🔢 Step 9: Computing Betti numbers..." << std::endl;
        for (int dim = 0; dim <= 2; ++dim) {
            int betti = persistent_cohomology.betti_number(dim);
            std::cout << "  H" << dim << " = " << betti << std::endl;
        }
        
        // Expected for triangle: H0=1 (connected), H1=0 (no holes), H2=0 (no voids)
        int h0 = persistent_cohomology.betti_number(0);
        int h1 = persistent_cohomology.betti_number(1);
        
        if (h0 == 1) {
            std::cout << "✅ H0=1 (connected) ✓" << std::endl;
        } else {
            std::cout << "❌ H0=" << h0 << " (expected 1)" << std::endl;
        }
        
        if (h1 == 0) {
            std::cout << "✅ H1=0 (no holes for triangle) ✓" << std::endl;
        } else {
            std::cout << "❌ H1=" << h1 << " (expected 0 for triangle)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 Direct GUDHI test completed!" << std::endl;
    return 0;
}

