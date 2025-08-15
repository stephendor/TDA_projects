#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>
#include <Eigen/Core>
#include <vector>

int main() { 
    // Test basic Eigen functionality - verify Eigen is working
    Eigen::MatrixXd m(2,2); 
    m << 1.0, 2.0, 3.0, 4.0;
    
    // Test that GUDHI headers can be included together with Eigen
    // Create a simple simplex tree (no complex instantiation)
    Gudhi::Simplex_tree<> simplex;
    
    // If we get here, the headers are compatible
    return 0; 
}
