#pragma once
#include "../core/types.hpp"
#include <vector>
#include <immintrin.h> // For AVX/SSE instructions

namespace tda::utils {

/**
 * @brief SIMD-optimized utilities for TDA operations
 * 
 * Provides vectorized implementations of common mathematical operations
 * using AVX2/SSE4.2 instructions for improved performance.
 */
class SIMDUtils {
public:
    // Check CPU capabilities at runtime
    struct CPUFeatures {
        bool has_avx = false;
        bool has_avx2 = false; 
        bool has_fma = false;
        bool has_avx512 = false;
    };
    
    static CPUFeatures detectCPUFeatures();
    
    /**
     * @brief Check if SIMD instructions are available
     * @return True if AVX2 is supported, false otherwise
     */
    static bool isAVX2Supported();
    
    /**
     * @brief Check if SSE4.2 instructions are available
     * @return True if SSE4.2 is supported, false otherwise
     */
    static bool isSSE42Supported();
    
    /**
     * @brief Vectorized Euclidean distance computation
     * @param a First point vector
     * @param b Second point vector
     * @return Euclidean distance between points
     */
    static double euclideanDistance(const std::vector<double>& a, 
                                   const std::vector<double>& b);
    
    /**
     * @brief Vectorized Euclidean distance computation (pointer interface)
     * @param p1 First point data pointer
     * @param p2 Second point data pointer
     * @param dimensions Number of dimensions
     * @return Euclidean distance between points
     */
    static double vectorizedEuclideanDistance(
        const double* p1, const double* p2, size_t dimensions
    );
    
    /**
     * @brief Vectorized batch distance computation
     * @param points Vector of points
     * @param query_point Query point
     * @return Vector of distances from query point to all points
     */
    static std::vector<double> computeDistancesBatch(
        const std::vector<std::vector<double>>& points,
        const std::vector<double>& query_point);
    
    /**
     * @brief Vectorized matrix-vector multiplication
     * @param matrix Input matrix (row-major)
     * @param vector Input vector
     * @return Result vector
     */
    static std::vector<double> matrixVectorMultiply(
        const std::vector<std::vector<double>>& matrix,
        const std::vector<double>& vector);
    
    /**
     * @brief Vectorized vector addition
     * @param a First vector
     * @param b Second vector
     * @return Sum vector
     */
    static std::vector<double> vectorAdd(const std::vector<double>& a,
                                        const std::vector<double>& b);
    
    /**
     * @brief Vectorized vector subtraction
     * @param a First vector
     * @param b Second vector
     * @return Difference vector
     */
    static std::vector<double> vectorSubtract(const std::vector<double>& a,
                                             const std::vector<double>& b);
    
    /**
     * @brief Vectorized vector dot product
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    static double vectorDotProduct(const std::vector<double>& a,
                                  const std::vector<double>& b);
    
    /**
     * @brief Vectorized vector normalization
     * @param vector Input vector
     * @return Normalized vector
     */
    static std::vector<double> vectorNormalize(const std::vector<double>& vector);
    
    /**
     * @brief Vectorized scalar-vector multiplication
     * @param scalar Scalar value
     * @param vector Input vector
     * @return Scaled vector
     */
    static std::vector<double> scalarVectorMultiply(double scalar,
                                                   const std::vector<double>& vector);
    
    // Vectorized operations for different data types
    static void vectorizedAdd(
        const double* a, const double* b, double* result, size_t count
    );
    
    static void vectorizedSubtract(
        const double* a, const double* b, double* result, size_t count  
    );
    
    static void vectorizedMultiply(
        const double* a, const double* b, double* result, size_t count
    );
    
    static void vectorizedDistanceMatrix(
        const std::vector<std::vector<double>>& points,
        double* distance_matrix
    );
    
    // Dot product and norms
    static double vectorizedDotProduct(
        const double* a, const double* b, size_t count
    );
    
    static double vectorizedL2Norm(const double* a, size_t count);

private:
    /**
     * @brief Align vector size to SIMD register width
     * @param size Original vector size
     * @return Aligned size for optimal SIMD processing
     */
    static size_t alignSize(size_t size);
    
    /**
     * @brief Process vector in SIMD chunks
     * @param data Pointer to data
     * @param size Size of data
     * @param func Function to apply to each SIMD chunk
     */
    template<typename Func>
    static void processSIMDChunks(const double* data, size_t size, Func func);
};

} // namespace tda::utils
