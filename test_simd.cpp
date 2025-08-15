#include "include/tda/utils/simd_utils.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "🧪 Testing SIMD Implementation..." << std::endl;
    
    // Check SIMD support
    std::cout << "AVX2 Support: " << (tda::utils::SIMDUtils::isAVX2Supported() ? "✅ Yes" : "❌ No") << std::endl;
    std::cout << "SSE4.2 Support: " << (tda::utils::SIMDUtils::isSSE42Supported() ? "✅ Yes" : "❌ No") << std::endl;
    
    // Create test vectors
    std::vector<double> a(1000, 1.0);
    std::vector<double> b(1000, 2.0);
    
    // Test vector operations
    std::cout << "\n🔬 Testing Vector Operations..." << std::endl;
    
    // Test vector addition
    auto start = std::chrono::high_resolution_clock::now();
    auto result = tda::utils::SIMDUtils::vectorAdd(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector Addition: " << duration.count() << " μs" << std::endl;
    std::cout << "Result[0]: " << result[0] << " (expected: 3.0)" << std::endl;
    
    // Test Euclidean distance
    start = std::chrono::high_resolution_clock::now();
    double distance = tda::utils::SIMDUtils::euclideanDistance(a, b);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Euclidean Distance: " << duration.count() << " μs" << std::endl;
    std::cout << "Distance: " << distance << " (expected: ~31.62)" << std::endl;
    
    // Test dot product
    start = std::chrono::high_resolution_clock::now();
    double dot_product = tda::utils::SIMDUtils::vectorDotProduct(a, b);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Dot Product: " << duration.count() << " μs" << std::endl;
    std::cout << "Dot Product: " << dot_product << " (expected: 2000.0)" << std::endl;
    
    std::cout << "\n✅ SIMD Test Completed Successfully!" << std::endl;
    return 0;
}
