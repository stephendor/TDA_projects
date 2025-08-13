#include <iostream>
#include "tda/vector_stack/vector_stack.hpp"

// Forward declaration
void run_algorithm_benchmarks();

void run_vector_stack_benchmarks() {
    std::cout << "TDA Vector Stack Benchmarks" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Vector stack benchmarks will be implemented here" << std::endl;
}

int main() {
    std::cout << "TDA Vector Stack - All Benchmarks" << std::endl;
    std::cout << "=================================" << std::endl;
    
    run_vector_stack_benchmarks();
    std::cout << std::endl;
    
    // Run algorithm benchmarks
    run_algorithm_benchmarks();
    
    return 0;
}
