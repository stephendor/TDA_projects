#include <iostream>
#include <vector>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Rips_complex.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/distance_functions.h>

int main() {
    std::cout << "🧪 Testing GUDHI Integration with C++23" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        // Test 1: Simplex Tree
        std::cout << "\n1️⃣ Testing Simplex Tree..." << std::endl;
        Gudhi::Simplex_tree<> simplex_tree;
        
        // Add some simplices
        simplex_tree.insert_simplex({0, 1}, 1.0);
        simplex_tree.insert_simplex({0, 2}, 1.0);
        simplex_tree.insert_simplex({1, 2}, 1.0);
        simplex_tree.insert_simplex({0, 1, 2}, 1.0);
        
        std::cout << "   Simplex tree created with " << simplex_tree.num_simplices() << " simplices" << std::endl;
        std::cout << "   Max dimension: " << simplex_tree.dimension() << std::endl;
        
        // Test 2: Rips Complex
        std::cout << "\n2️⃣ Testing Rips Complex..." << std::endl;
        std::vector<std::vector<double>> points = {
            {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
        };
        
        // Use the correct API based on examples
        using Simplex_tree = Gudhi::Simplex_tree<>;
        using Filtration_value = Simplex_tree::Filtration_value;
        using Rips_complex = Gudhi::rips_complex::Rips_complex<Filtration_value>;
        
        // Create a new simplex tree for the Rips complex
        Simplex_tree rips_simplex_tree;
        Rips_complex rips_complex(points, 2.0, Gudhi::Euclidean_distance());
        rips_complex.create_complex(rips_simplex_tree, 3);
        
        std::cout << "   Rips complex created with " << rips_simplex_tree.num_simplices() << " simplices" << std::endl;
        
        // Test 3: Persistent Cohomology
        std::cout << "\n3️⃣ Testing Persistent Cohomology..." << std::endl;
        using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
        
        Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp> pcoh(rips_simplex_tree);
        pcoh.init_coefficients(2); // Z/2Z coefficients
        pcoh.compute_persistent_cohomology();
        
        std::cout << "   Persistent cohomology computed successfully" << std::endl;
        
        // Test 4: Basic TDA Operations
        std::cout << "\n4️⃣ Testing Basic TDA Operations..." << std::endl;
        
        // Initialize filtration
        rips_simplex_tree.initialize_filtration();
        
        // Count simplices by dimension
        std::vector<int> simplex_count_by_dim(rips_simplex_tree.dimension() + 1, 0);
        for (auto sh : rips_simplex_tree.filtration_simplex_range()) {
            simplex_count_by_dim[rips_simplex_tree.dimension(sh)]++;
        }
        
        std::cout << "   Simplex count by dimension:" << std::endl;
        for (size_t i = 0; i < simplex_count_by_dim.size(); ++i) {
            std::cout << "     Dimension " << i << ": " << simplex_count_by_dim[i] << " simplices" << std::endl;
        }
        
        // Test 5: Distance Computations
        std::cout << "\n5️⃣ Testing Distance Computations..." << std::endl;
        auto distance = Gudhi::Euclidean_distance();
        double dist_01 = distance(points[0], points[1]);
        double dist_02 = distance(points[0], points[2]);
        double dist_03 = distance(points[0], points[3]);
        
        std::cout << "   Distance (0,1): " << dist_01 << std::endl;
        std::cout << "   Distance (0,2): " << dist_02 << std::endl;
        std::cout << "   Distance (0,3): " << dist_03 << std::endl;
        
        std::cout << "\n✅ GUDHI integration successful!" << std::endl;
        std::cout << "🚀 Ready for TDA algorithm development!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
