#include "../../../include/tda/utils/simd_utils.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tda::utils {

bool SIMDUtils::isAVX2Supported() {
    #ifdef __AVX2__
        return true;
    #else
        return false;
    #endif
}

bool SIMDUtils::isSSE42Supported() {
    #ifdef __SSE4_2__
        return true;
    #else
        return false;
    #endif
}

double SIMDUtils::euclideanDistance(const std::vector<double>& a, 
                                   const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for distance computation");
    }
    
    if (a.empty()) {
        return 0.0;
    }
    
    const size_t size = a.size();
    const double* a_ptr = a.data();
    const double* b_ptr = b.data();
    
    double sum = 0.0;
    
    // Use AVX2 if available for optimal performance
    if (isAVX2Supported() && size >= 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        const size_t aligned_size = size - (size % 4);
        
        // Process 4 doubles at a time with AVX2
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a_ptr[i]);
            __m256d b_vec = _mm256_loadu_pd(&b_ptr[i]);
            __m256d diff = _mm256_sub_pd(a_vec, b_vec);
            __m256d diff_squared = _mm256_mul_pd(diff, diff);
            sum_vec = _mm256_add_pd(sum_vec, diff_squared);
        }
        
        // Extract sum from vector
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < size; ++i) {
            double diff = a_ptr[i] - b_ptr[i];
            sum += diff * diff;
        }
    }
    // Fallback to SSE4.2 if AVX2 not available
    else if (isSSE42Supported() && size >= 2) {
        __m128d sum_vec = _mm_setzero_pd();
        const size_t aligned_size = size - (size % 2);
        
        // Process 2 doubles at a time with SSE4.2
        for (size_t i = 0; i < aligned_size; i += 2) {
            __m128d a_vec = _mm_loadu_pd(&a_ptr[i]);
            __m128d b_vec = _mm_loadu_pd(&b_ptr[i]);
            __m128d diff = _mm_sub_pd(a_vec, b_vec);
            __m128d diff_squared = _mm_mul_pd(diff, diff);
            sum_vec = _mm_add_pd(sum_vec, diff_squared);
        }
        
        // Extract sum from vector
        double temp[2];
        _mm_storeu_pd(temp, sum_vec);
        sum = temp[0] + temp[1];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < size; ++i) {
            double diff = a_ptr[i] - b_ptr[i];
            sum += diff * diff;
        }
    }
    // Fallback to scalar implementation
    else {
        for (size_t i = 0; i < size; ++i) {
            double diff = a_ptr[i] - b_ptr[i];
            sum += diff * diff;
        }
    }
    
    return std::sqrt(sum);
}

std::vector<double> SIMDUtils::computeDistancesBatch(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& query_point) {
    
    if (points.empty()) {
        return {};
    }
    
    std::vector<double> distances;
    distances.reserve(points.size());
    
    for (const auto& point : points) {
        distances.push_back(euclideanDistance(point, query_point));
    }
    
    return distances;
}

std::vector<double> SIMDUtils::matrixVectorMultiply(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>& vector) {
    
    if (matrix.empty() || vector.empty()) {
        return {};
    }
    
    const size_t rows = matrix.size();
    const size_t cols = vector.size();
    
    // Check if matrix is rectangular
    for (const auto& row : matrix) {
        if (row.size() != cols) {
            throw std::invalid_argument("Matrix must be rectangular");
        }
    }
    
    std::vector<double> result(rows, 0.0);
    
    // Use SIMD for matrix-vector multiplication if available
    if (isAVX2Supported() && cols >= 4) {
        for (size_t i = 0; i < rows; ++i) {
            __m256d sum_vec = _mm256_setzero_pd();
            const size_t aligned_cols = cols - (cols % 4);
            
            // Process 4 doubles at a time
            for (size_t j = 0; j < aligned_cols; j += 4) {
                __m256d matrix_row = _mm256_loadu_pd(&matrix[i][j]);
                __m256d vector_chunk = _mm256_loadu_pd(&vector[j]);
                __m256d product = _mm256_mul_pd(matrix_row, vector_chunk);
                sum_vec = _mm256_add_pd(sum_vec, product);
            }
            
            // Extract sum from vector
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            result[i] = temp[0] + temp[1] + temp[2] + temp[3];
            
            // Handle remaining elements
            for (size_t j = aligned_cols; j < cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
    }
    // Fallback to scalar implementation
    else {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
    }
    
    return result;
}

std::vector<double> SIMDUtils::vectorAdd(const std::vector<double>& a,
                                        const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }
    
    if (a.empty()) {
        return {};
    }
    
    std::vector<double> result(a.size());
    const double* a_ptr = a.data();
    const double* b_ptr = b.data();
    double* result_ptr = result.data();
    const size_t size = a.size();
    
    // Use AVX2 if available
    if (isAVX2Supported() && size >= 4) {
        const size_t aligned_size = size - (size % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a_ptr[i]);
            __m256d b_vec = _mm256_loadu_pd(&b_ptr[i]);
            __m256d sum_vec = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result_ptr[i], sum_vec);
        }
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < size; ++i) {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }
    // Fallback to scalar
    else {
        for (size_t i = 0; i < size; ++i) {
            result_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    }
    
    return result;
}

std::vector<double> SIMDUtils::vectorSubtract(const std::vector<double>& a,
                                             const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for subtraction");
    }
    
    if (a.empty()) {
        return {};
    }
    
    std::vector<double> result(a.size());
    const double* a_ptr = a.data();
    const double* b_ptr = b.data();
    double* result_ptr = result.data();
    const size_t size = a.size();
    
    // Use AVX2 if available
    if (isAVX2Supported() && size >= 4) {
        const size_t aligned_size = size - (size % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a_ptr[i]);
            __m256d b_vec = _mm256_loadu_pd(&b_ptr[i]);
            __m256d diff_vec = _mm256_sub_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result_ptr[i], diff_vec);
        }
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < size; ++i) {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
        }
    }
    // Fallback to scalar
    else {
        for (size_t i = 0; i < size; ++i) {
            result_ptr[i] = a_ptr[i] - b_ptr[i];
        }
    }
    
    return result;
}

double SIMDUtils::vectorDotProduct(const std::vector<double>& a,
                                  const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }
    
    if (a.empty()) {
        return 0.0;
    }
    
    const double* a_ptr = a.data();
    const double* b_ptr = b.data();
    const size_t size = a.size();
    double sum = 0.0;
    
    // Use AVX2 if available
    if (isAVX2Supported() && size >= 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        const size_t aligned_size = size - (size % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a_ptr[i]);
            __m256d b_vec = _mm256_loadu_pd(&b_ptr[i]);
            __m256d product = _mm256_mul_pd(a_vec, b_vec);
            sum_vec = _mm256_add_pd(sum_vec, product);
        }
        
        // Extract sum from vector
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < size; ++i) {
            sum += a_ptr[i] * b_ptr[i];
        }
    }
    // Fallback to scalar
    else {
        for (size_t i = 0; i < size; ++i) {
            sum += a_ptr[i] * b_ptr[i];
        }
    }
    
    return sum;
}

std::vector<double> SIMDUtils::vectorNormalize(const std::vector<double>& vector) {
    if (vector.empty()) {
        return {};
    }
    
    double magnitude = std::sqrt(vectorDotProduct(vector, vector));
    if (magnitude == 0.0) {
        return vector; // Return original vector if magnitude is zero
    }
    
    return scalarVectorMultiply(1.0 / magnitude, vector);
}

std::vector<double> SIMDUtils::scalarVectorMultiply(double scalar,
                                                   const std::vector<double>& vector) {
    if (vector.empty()) {
        return {};
    }
    
    std::vector<double> result(vector.size());
    const double* vector_ptr = vector.data();
    double* result_ptr = result.data();
    const size_t size = vector.size();
    
    // Use AVX2 if available
    if (isAVX2Supported() && size >= 4) {
        __m256d scalar_vec = _mm256_set1_pd(scalar);
        const size_t aligned_size = size - (size % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d vector_chunk = _mm256_loadu_pd(&vector_ptr[i]);
            __m256d product = _mm256_mul_pd(scalar_vec, vector_chunk);
            _mm256_storeu_pd(&result_ptr[i], product);
        }
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < size; ++i) {
            result_ptr[i] = scalar * vector_ptr[i];
        }
    }
    // Fallback to scalar
    else {
        for (size_t i = 0; i < size; ++i) {
            result_ptr[i] = scalar * vector_ptr[i];
        }
    }
    
    return result;
}

SIMDUtils::CPUFeatures SIMDUtils::detectCPUFeatures() {
    CPUFeatures features;
    
    #ifdef __AVX__
        features.has_avx = true;
    #endif
    #ifdef __AVX2__
        features.has_avx2 = true;
    #endif
    #ifdef __FMA__
        features.has_fma = true;
    #endif
    #ifdef __AVX512F__
        features.has_avx512 = true;
    #endif
    
    return features;
}

double SIMDUtils::vectorizedEuclideanDistance(
    const double* p1, const double* p2, size_t dimensions
) {
    if (!p1 || !p2 || dimensions == 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    
    // Use AVX2 if available for optimal performance
    if (isAVX2Supported() && dimensions >= 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        const size_t aligned_size = dimensions - (dimensions % 4);
        
        // Process 4 doubles at a time with AVX2
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&p1[i]);
            __m256d b_vec = _mm256_loadu_pd(&p2[i]);
            __m256d diff = _mm256_sub_pd(a_vec, b_vec);
            
            #ifdef __FMA__
                sum_vec = _mm256_fmadd_pd(diff, diff, sum_vec);
            #else
                __m256d diff_squared = _mm256_mul_pd(diff, diff);
                sum_vec = _mm256_add_pd(sum_vec, diff_squared);
            #endif
        }
        
        // Extract sum from vector
        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < dimensions; ++i) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
    }
    // Fallback to SSE4.2 if AVX2 not available
    else if (isSSE42Supported() && dimensions >= 2) {
        __m128d sum_vec = _mm_setzero_pd();
        const size_t aligned_size = dimensions - (dimensions % 2);
        
        // Process 2 doubles at a time with SSE4.2
        for (size_t i = 0; i < aligned_size; i += 2) {
            __m128d a_vec = _mm_loadu_pd(&p1[i]);
            __m128d b_vec = _mm_loadu_pd(&p2[i]);
            __m128d diff = _mm_sub_pd(a_vec, b_vec);
            __m128d diff_squared = _mm_mul_pd(diff, diff);
            sum_vec = _mm_add_pd(sum_vec, diff_squared);
        }
        
        // Extract sum from vector
        alignas(16) double temp[2];
        _mm_store_pd(temp, sum_vec);
        sum = temp[0] + temp[1];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < dimensions; ++i) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
    }
    // Fallback to scalar implementation
    else {
        for (size_t i = 0; i < dimensions; ++i) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
    }
    
    return std::sqrt(sum);
}

void SIMDUtils::vectorizedAdd(
    const double* a, const double* b, double* result, size_t count
) {
    if (!a || !b || !result || count == 0) return;
    
    if (isAVX2Supported() && count >= 4) {
        const size_t aligned_size = count - (count % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result[i], result_vec);
        }
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < count; ++i) {
            result[i] = a[i] + b[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

void SIMDUtils::vectorizedSubtract(
    const double* a, const double* b, double* result, size_t count
) {
    if (!a || !b || !result || count == 0) return;
    
    if (isAVX2Supported() && count >= 4) {
        const size_t aligned_size = count - (count % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d result_vec = _mm256_sub_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result[i], result_vec);
        }
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < count; ++i) {
            result[i] = a[i] - b[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            result[i] = a[i] - b[i];
        }
    }
}

void SIMDUtils::vectorizedMultiply(
    const double* a, const double* b, double* result, size_t count
) {
    if (!a || !b || !result || count == 0) return;
    
    if (isAVX2Supported() && count >= 4) {
        const size_t aligned_size = count - (count % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d result_vec = _mm256_mul_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result[i], result_vec);
        }
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < count; ++i) {
            result[i] = a[i] * b[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            result[i] = a[i] * b[i];
        }
    }
}

double SIMDUtils::vectorizedDotProduct(
    const double* a, const double* b, size_t count
) {
    if (!a || !b || count == 0) return 0.0;
    
    double sum = 0.0;
    
    if (isAVX2Supported() && count >= 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        const size_t aligned_size = count - (count % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            
            #ifdef __FMA__
                sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
            #else
                __m256d product = _mm256_mul_pd(a_vec, b_vec);
                sum_vec = _mm256_add_pd(sum_vec, product);
            #endif
        }
        
        // Extract sum from vector
        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < count; ++i) {
            sum += a[i] * b[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            sum += a[i] * b[i];
        }
    }
    
    return sum;
}

double SIMDUtils::vectorizedL2Norm(const double* a, size_t count) {
    if (!a || count == 0) return 0.0;
    
    double sum = 0.0;
    
    if (isAVX2Supported() && count >= 4) {
        __m256d sum_vec = _mm256_setzero_pd();
        const size_t aligned_size = count - (count % 4);
        
        for (size_t i = 0; i < aligned_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            
            #ifdef __FMA__
                sum_vec = _mm256_fmadd_pd(a_vec, a_vec, sum_vec);
            #else
                __m256d squared = _mm256_mul_pd(a_vec, a_vec);
                sum_vec = _mm256_add_pd(sum_vec, squared);
            #endif
        }
        
        // Extract sum from vector
        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum_vec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Handle remaining elements
        for (size_t i = aligned_size; i < count; ++i) {
            sum += a[i] * a[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            sum += a[i] * a[i];
        }
    }
    
    return std::sqrt(sum);
}

void SIMDUtils::vectorizedDistanceMatrix(
    const std::vector<std::vector<double>>& points,
    double* distance_matrix
) {
    if (points.empty() || !distance_matrix) return;
    
    const size_t n = points.size();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = vectorizedEuclideanDistance(
                points[i].data(), points[j].data(), points[i].size()
            );
            distance_matrix[i * n + j] = dist;
            distance_matrix[j * n + i] = dist; // Symmetric matrix
        }
        distance_matrix[i * n + i] = 0.0; // Diagonal is zero
    }
}

size_t SIMDUtils::alignSize(size_t size) {
    // Align to 4 doubles (32 bytes) for AVX2
    if (isAVX2Supported()) {
        return (size + 3) & ~3;
    }
    // Align to 2 doubles (16 bytes) for SSE4.2
    else if (isSSE42Supported()) {
        return (size + 1) & ~1;
    }
    // No alignment needed for scalar
    return size;
}

} // namespace tda::utils
