#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <cmath>
#include "tda/utils/streaming_distance_cuda.hpp"

namespace tda::utils {

static __global__ void distance_tile_kernel(
    const double* __restrict__ points,
    size_t stride, // dim
    const size_t* __restrict__ row_idx,
    const size_t* __restrict__ col_idx,
    size_t i_len,
    size_t j_len,
    size_t dim,
    double* __restrict__ out)
{
    size_t ii = blockIdx.y * blockDim.y + threadIdx.y;
    size_t jj = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii >= i_len || jj >= j_len) return;
    size_t i = row_idx[ii];
    size_t j = col_idx[jj];
    const double* a = points + i * stride;
    const double* b = points + j * stride;
    double s = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        double d = a[k] - b[k];
        s += d * d;
    }
    out[ii * j_len + jj] = sqrt(s);
}

bool compute_distance_tile_cuda(
    const std::vector<std::vector<double>>& points,
    size_t bi,
    size_t bj,
    size_t i_len,
    size_t j_len,
    size_t dim,
    std::vector<std::vector<double>>& tile)
{
    // Flatten points
    const size_t n = points.size();
    if (n == 0 || dim == 0 || i_len == 0 || j_len == 0) return false;
    std::vector<double> h_points(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < dim; ++k) h_points[i * dim + k] = points[i][k];
    }
    // Build row/col index arrays
    std::vector<size_t> h_row(i_len), h_col(j_len);
    for (size_t r = 0; r < i_len; ++r) h_row[r] = bi + r;
    for (size_t c = 0; c < j_len; ++c) h_col[c] = bj + c;

    double* d_points = nullptr; size_t* d_row = nullptr; size_t* d_col = nullptr; double* d_out = nullptr;
    cudaError_t err;
    if ((err = cudaMalloc(&d_points, h_points.size() * sizeof(double))) != cudaSuccess) return false;
    if ((err = cudaMalloc(&d_row, i_len * sizeof(size_t))) != cudaSuccess) { cudaFree(d_points); return false; }
    if ((err = cudaMalloc(&d_col, j_len * sizeof(size_t))) != cudaSuccess) { cudaFree(d_points); cudaFree(d_row); return false; }
    if ((err = cudaMalloc(&d_out, i_len * j_len * sizeof(double))) != cudaSuccess) { cudaFree(d_points); cudaFree(d_row); cudaFree(d_col); return false; }

    if ((err = cudaMemcpy(d_points, h_points.data(), h_points.size() * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_out); cudaFree(d_col); cudaFree(d_row); cudaFree(d_points); return false; }
    if ((err = cudaMemcpy(d_row, h_row.data(), i_len * sizeof(size_t), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_out); cudaFree(d_col); cudaFree(d_row); cudaFree(d_points); return false; }
    if ((err = cudaMemcpy(d_col, h_col.data(), j_len * sizeof(size_t), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_out); cudaFree(d_col); cudaFree(d_row); cudaFree(d_points); return false; }

    dim3 block(16, 16);
    dim3 grid((j_len + block.x - 1) / block.x, (i_len + block.y - 1) / block.y);
    distance_tile_kernel<<<grid, block>>>(d_points, dim, d_row, d_col, i_len, j_len, dim, d_out);
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) { cudaFree(d_out); cudaFree(d_col); cudaFree(d_row); cudaFree(d_points); return false; }

    std::vector<double> h_out;
    h_out.resize(i_len * j_len);
    if ((err = cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess) { cudaFree(d_out); cudaFree(d_col); cudaFree(d_row); cudaFree(d_points); return false; }
    // Copy back to 2D tile
    for (size_t ii = 0; ii < i_len; ++ii) {
        for (size_t jj = 0; jj < j_len; ++jj) {
            tile[ii][jj] = h_out[ii * j_len + jj];
        }
    }
    cudaFree(d_out); cudaFree(d_col); cudaFree(d_row); cudaFree(d_points);
    return true;

    // Unreachable cleanup label removed (early returns used above)
}

} // namespace tda::utils



