#pragma once

#include <vector>
#include <cstddef>

// CUDA-accelerated distance tile computation (optional)
// Guard all declarations with TDA_ENABLE_CUDA in callers.
namespace tda::utils {

// Attempts to compute an i_len x j_len distance tile on GPU for the submatrix
// defined by rows [bi, bi + i_len) and columns [bj, bj + j_len) of the input
// point set. Points are given as vector-of-vector with uniform dimension 'dim'.
// The output 'tile' must be sized to [i_len][j_len]. Returns true on success,
// false if CUDA is unavailable or a runtime error occurred (callers should
// fall back to CPU in that case).
bool compute_distance_tile_cuda(
    const std::vector<std::vector<double>>& points,
    size_t bi,
    size_t bj,
    size_t i_len,
    size_t j_len,
    size_t dim,
    std::vector<std::vector<double>>& tile);

} // namespace tda::utils




