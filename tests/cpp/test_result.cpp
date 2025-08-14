#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include "tda/core/types.hpp"

namespace {

void test_types_basic() {
    std::cout << "Testing basic types..." << std::endl;
    
    // Test Vector types
    tda::core::Vector3D point{1.0, 2.0, 3.0};
    assert(point.size() == 3);
    assert(point[0] == 1.0);
    assert(point[1] == 2.0);
    assert(point[2] == 3.0);
    
    // Test DynamicVector
    tda::core::DynamicVector vec{4.0, 5.0, 6.0, 7.0};
    assert(vec.size() == 4);
    assert(vec[0] == 4.0);
    
    std::cout << "✅ Basic types test passed" << std::endl;
}

void test_persistence_pair() {
    std::cout << "Testing PersistencePair type..." << std::endl;
    
    tda::core::PersistencePair pair(1, 2.5, 3.8, 0, 1);
    assert(pair.birth == 2.5);
    assert(pair.death == 3.8);
    assert(pair.dimension == 1);
    // Allow for floating-point rounding differences
    assert(std::abs(pair.get_persistence() - 1.3) < 1e-12);
    
    std::cout << "✅ PersistencePair test passed" << std::endl;
}

void test_betti_numbers() {
    std::cout << "Testing BettiNumbers type..." << std::endl;
    
    tda::core::BettiNumbers betti(3);
    betti[0] = 1;
    betti[1] = 2;
    betti[2] = 1;
    betti[3] = 0;
    
    assert(betti.max_dimension() == 3);
    assert(betti.total_betti() == 4);
    
    std::cout << "✅ BettiNumbers test passed" << std::endl;
}

void test_result_type() {
    std::cout << "Testing Result type..." << std::endl;
    
    // Test success case
    auto success_result = tda::core::Result<int>::success(42);
    assert(success_result.has_value());
    assert(!success_result.has_error());
    assert(success_result.value() == 42);
    
    // Test error case
    auto error_result = tda::core::Result<int>::failure("Test error");
    assert(!error_result.has_value());
    assert(error_result.has_error());
    assert(error_result.error() == "Test error");
    
    std::cout << "✅ Result type test passed" << std::endl;
}

} // anonymous namespace

int main() {
    std::cout << "🧪 Running Core Types Tests" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        test_types_basic();
        test_persistence_pair();
        test_betti_numbers();
        test_result_type();
        
        std::cout << std::endl;
        std::cout << "🎉 All Types tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
