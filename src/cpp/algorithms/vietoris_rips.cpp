#include "../../../include/tda/core/types.hpp"
#include "../../../include/tda/core/simplex.hpp"
#include "../../../include/tda/core/point_cloud.hpp"
#include <algorithm>
#include <execution>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>

namespace tda::algorithms {

class VietorisRipsComplex {
private:
    std::vector<core::Simplex> simplices_;
    std::vector<double> birth_times_;
    std::unordered_map<core::Index, std::vector<core::Index>> adjacency_list_;
    double max_radius_;
    core::Dimension max_dimension_;
    
public:
    // Type aliases
    using size_type = std::size_t;
    using iterator = std::vector<core::Simplex>::iterator;
    using const_iterator = std::vector<core::Simplex>::const_iterator;
    
    // Constructors
    VietorisRipsComplex() = default;
    
    VietorisRipsComplex(double max_radius, core::Dimension max_dimension = 3)
        : max_radius_(max_radius), max_dimension_(max_dimension) {}
    
    // Copy constructor
    VietorisRipsComplex(const VietorisRipsComplex& other)
        : simplices_(other.simplices_), birth_times_(other.birth_times_),
          adjacency_list_(other.adjacency_list_), max_radius_(other.max_radius_),
          max_dimension_(other.max_dimension_) {}
    
    // Move constructor
    VietorisRipsComplex(VietorisRipsComplex&& other) noexcept
        : simplices_(std::move(other.simplices_)), birth_times_(std::move(other.birth_times_)),
          adjacency_list_(std::move(other.adjacency_list_)), max_radius_(other.max_radius_),
          max_dimension_(other.max_dimension_) {
        other.max_radius_ = 0.0;
        other.max_dimension_ = 0;
    }
    
    // Destructor
    ~VietorisRipsComplex() = default;
    
    // Assignment operators
    VietorisRipsComplex& operator=(const VietorisRipsComplex& other) {
        if (this != &other) {
            simplices_ = other.simplices_;
            birth_times_ = other.birth_times_;
            adjacency_list_ = other.adjacency_list_;
            max_radius_ = other.max_radius_;
            max_dimension_ = other.max_dimension_;
        }
        return *this;
    }
    
    VietorisRipsComplex& operator=(VietorisRipsComplex&& other) noexcept {
        if (this != &other) {
            simplices_ = std::move(other.simplices_);
            birth_times_ = std::move(other.birth_times_);
            adjacency_list_ = std::move(other.adjacency_list_);
            max_radius_ = other.max_radius_;
            max_dimension_ = other.max_dimension_;
            other.max_radius_ = 0.0;
            other.max_dimension_ = 0;
        }
        return *this;
    }
    
    // Configuration
    void set_max_radius(double radius) noexcept { max_radius_ = radius; }
    void set_max_dimension(core::Dimension dim) noexcept { max_dimension_ = dim; }
    
    double max_radius() const noexcept { return max_radius_; }
    core::Dimension max_dimension() const noexcept { return max_dimension_; }
    
    // Data access
    const std::vector<core::Simplex>& simplices() const noexcept { return simplices_; }
    std::vector<core::Simplex>& simplices() noexcept { return simplices_; }
    
    const std::vector<double>& birth_times() const noexcept { return birth_times_; }
    std::vector<double>& birth_times() noexcept { return birth_times_; }
    
    // Iterators
    iterator begin() noexcept { return simplices_.begin(); }
    const_iterator begin() const noexcept { return simplices_.begin(); }
    const_iterator cbegin() const noexcept { return simplices_.cbegin(); }
    
    iterator end() noexcept { return simplices_.end(); }
    const_iterator end() const noexcept { return simplices_.end(); }
    const_iterator cend() const noexcept { return simplices_.cend(); }
    
    // Capacity
    size_type size() const noexcept { return simplices_.size(); }
    bool empty() const noexcept { return simplices_.empty(); }
    
    // Complex construction
    void build_from_point_cloud(const core::PointCloud& point_cloud) {
        clear();
        
        size_t n = point_cloud.size();
        if (n == 0) return;
        
        // Add vertices (0-simplices)
        for (size_t i = 0; i < n; ++i) {
            simplices_.emplace_back(std::vector<core::Index>{static_cast<core::Index>(i)});
            birth_times_.push_back(0.0);
        }
        
        // Compute pairwise distances and build adjacency list
        std::vector<std::pair<double, std::pair<size_t, size_t>>> edges;
        edges.reserve(n * (n - 1) / 2);
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double distance = point_cloud.compute_distance(i, j);
                if (distance <= max_radius_) {
                    edges.emplace_back(distance, std::make_pair(i, j));
                    adjacency_list_[i].push_back(j);
                    adjacency_list_[j].push_back(i);
                }
            }
        }
        
        // Sort edges by distance
        std::sort(std::execution::par_unseq, edges.begin(), edges.end());
        
        // Add edges (1-simplices)
        for (const auto& edge : edges) {
            size_t i = edge.second.first;
            size_t j = edge.second.second;
            simplices_.emplace_back(std::vector<core::Index>{
                static_cast<core::Index>(i), static_cast<core::Index>(j)
            });
            birth_times_.push_back(edge.first);
        }
        
        // Build higher-dimensional simplices
        if (max_dimension_ >= 2) {
            build_triangles();
        }
        
        if (max_dimension_ >= 3) {
            build_tetrahedra();
        }
        
        // Sort simplices by birth time
        sort_by_birth_time();
    }
    
    void build_from_distance_matrix(const std::vector<std::vector<double>>& distance_matrix) {
        clear();
        
        size_t n = distance_matrix.size();
        if (n == 0) return;
        
        // Add vertices (0-simplices)
        for (size_t i = 0; i < n; ++i) {
            simplices_.emplace_back(std::vector<core::Index>{static_cast<core::Index>(i)});
            birth_times_.push_back(0.0);
        }
        
        // Build edges and adjacency list
        std::vector<std::pair<double, std::pair<size_t, size_t>>> edges;
        edges.reserve(n * (n - 1) / 2);
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double distance = distance_matrix[i][j];
                if (distance <= max_radius_) {
                    edges.emplace_back(distance, std::make_pair(i, j));
                    adjacency_list_[i].push_back(j);
                    adjacency_list_[j].push_back(i);
                }
            }
        }
        
        // Sort edges by distance
        std::sort(std::execution::par_unseq, edges.begin(), edges.end());
        
        // Add edges (1-simplices)
        for (const auto& edge : edges) {
            size_t i = edge.second.first;
            size_t j = edge.second.second;
            simplices_.emplace_back(std::vector<core::Index>{
                static_cast<core::Index>(i), static_cast<core::Index>(j)
            });
            birth_times_.push_back(edge.first);
        }
        
        // Build higher-dimensional simplices
        if (max_dimension_ >= 2) {
            build_triangles();
        }
        
        if (max_dimension_ >= 3) {
            build_tetrahedra();
        }
        
        // Sort simplices by birth time
        sort_by_birth_time();
    }
    
    // Filtration operations
    void sort_by_birth_time() {
        if (simplices_.size() != birth_times_.size()) {
            throw std::runtime_error("Simplex and birth time arrays have different sizes");
        }
        
        // Create index vector for sorting
        std::vector<size_type> indices(simplices_.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort indices by birth time
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(),
                  [this](size_type i, size_type j) {
                      return birth_times_[i] < birth_times_[j];
                  });
        
        // Reorder data according to sorted indices
        reorder_by_indices(indices);
    }
    
    void sort_by_dimension() {
        if (simplices_.size() != birth_times_.size()) {
            throw std::runtime_error("Simplex and birth time arrays have different sizes");
        }
        
        // Create index vector for sorting
        std::vector<size_type> indices(simplices_.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort indices by dimension, then by birth time
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(),
                  [this](size_type i, size_type j) {
                      if (simplices_[i].dimension() != simplices_[j].dimension()) {
                          return simplices_[i].dimension() < simplices_[j].dimension();
                      }
                      return birth_times_[i] < birth_times_[j];
                  });
        
        // Reorder data according to sorted indices
        reorder_by_indices(indices);
    }
    
    // Query operations
    std::vector<size_type> find_simplices_by_dimension(core::Dimension dim) const {
        std::vector<size_type> indices;
        indices.reserve(simplices_.size());
        
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (simplices_[i].dimension() == dim) {
                indices.push_back(i);
            }
        }
        
        return indices;
    }
    
    std::vector<size_type> find_simplices_in_radius_range(double min_radius, double max_radius) const {
        std::vector<size_type> indices;
        indices.reserve(simplices_.size());
        
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (birth_times_[i] >= min_radius && birth_times_[i] <= max_radius) {
                indices.push_back(i);
            }
        }
        
        return indices;
    }
    
    // Utility methods
    void clear() noexcept {
        simplices_.clear();
        birth_times_.clear();
        adjacency_list_.clear();
    }
    
    void reserve(size_type capacity) {
        simplices_.reserve(capacity);
        birth_times_.reserve(capacity);
    }
    
    void shrink_to_fit() {
        simplices_.shrink_to_fit();
        birth_times_.shrink_to_fit();
    }
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(simplices_, birth_times_, adjacency_list_, max_radius_, max_dimension_);
    }
    
private:
    // Helper method to get edge birth time
    double get_edge_birth_time(core::Index v1, core::Index v2) const {
        // Find the edge in the simplices list
        for (size_type i = 0; i < simplices_.size(); ++i) {
            const auto& simplex = simplices_[i];
            if (simplex.dimension() == 1 && simplex.size() == 2) {
                const auto& vertices = simplex.vertices();
                if ((vertices[0] == v1 && vertices[1] == v2) ||
                    (vertices[0] == v2 && vertices[1] == v1)) {
                    return birth_times_[i];
                }
            }
        }
        // If edge not found, return infinity (should not happen in valid complex)
        return std::numeric_limits<double>::infinity();
    }
    void build_triangles() {
        std::vector<std::tuple<double, core::Index, core::Index, core::Index>> triangles;
        
        // Find all triangles
        for (const auto& [vertex, neighbors] : adjacency_list_) {
            for (core::Index neighbor1 : neighbors) {
                if (neighbor1 <= vertex) continue;
                
                for (core::Index neighbor2 : neighbors) {
                    if (neighbor2 <= neighbor1) continue;
                    
                    // Check if neighbor1 and neighbor2 are connected
                    auto it = std::find(adjacency_list_[neighbor1].begin(), 
                                       adjacency_list_[neighbor1].end(), neighbor2);
                    if (it != adjacency_list_[neighbor1].end()) {
                        // Found a triangle - calculate birth time as max of its three edges
                        double edge1_time = get_edge_birth_time(vertex, neighbor1);
                        double edge2_time = get_edge_birth_time(vertex, neighbor2);
                        double edge3_time = get_edge_birth_time(neighbor1, neighbor2);
                        
                        double max_edge = std::max({edge1_time, edge2_time, edge3_time});
                        
                        triangles.emplace_back(max_edge, vertex, neighbor1, neighbor2);
                    }
                }
            }
        }
        
        // Sort triangles by birth time
        std::sort(std::execution::par_unseq, triangles.begin(), triangles.end());
        
        // Add triangles
        for (const auto& [birth_time, v1, v2, v3] : triangles) {
            simplices_.emplace_back(std::vector<core::Index>{v1, v2, v3});
            birth_times_.push_back(birth_time);
        }
    }
    
    void build_tetrahedra() {
        std::vector<std::tuple<double, core::Index, core::Index, core::Index, core::Index>> tetrahedra;
        
        // Find all tetrahedra
        for (const auto& [vertex, neighbors] : adjacency_list_) {
            for (core::Index neighbor1 : neighbors) {
                if (neighbor1 <= vertex) continue;
                
                for (core::Index neighbor2 : neighbors) {
                    if (neighbor2 <= neighbor1) continue;
                    
                    for (core::Index neighbor3 : neighbors) {
                        if (neighbor3 <= neighbor2) continue;
                        
                        // Check if all edges exist
                        if (is_triangle(neighbor1, neighbor2, neighbor3)) {
                            // Calculate birth time as max of all 6 edges of the tetrahedron
                            double edge1_time = get_edge_birth_time(vertex, neighbor1);
                            double edge2_time = get_edge_birth_time(vertex, neighbor2);
                            double edge3_time = get_edge_birth_time(vertex, neighbor3);
                            double edge4_time = get_edge_birth_time(neighbor1, neighbor2);
                            double edge5_time = get_edge_birth_time(neighbor1, neighbor3);
                            double edge6_time = get_edge_birth_time(neighbor2, neighbor3);
                            
                            double max_edge = std::max({edge1_time, edge2_time, edge3_time, 
                                                       edge4_time, edge5_time, edge6_time});
                            
                            tetrahedra.emplace_back(max_edge, vertex, neighbor1, neighbor2, neighbor3);
                        }
                    }
                }
            }
        }
        
        // Sort tetrahedra by birth time
        std::sort(std::execution::par_unseq, tetrahedra.begin(), tetrahedra.end());
        
        // Add tetrahedra
        for (const auto& [birth_time, v1, v2, v3, v4] : tetrahedra) {
            simplices_.emplace_back(std::vector<core::Index>{v1, v2, v3, v4});
            birth_times_.push_back(birth_time);
        }
    }
    
    bool is_triangle(core::Index v1, core::Index v2, core::Index v3) const {
        // Check if all three edges exist
        auto it1 = std::find(adjacency_list_.at(v1).begin(), adjacency_list_.at(v1).end(), v2);
        auto it2 = std::find(adjacency_list_.at(v2).begin(), adjacency_list_.at(v2).end(), v3);
        auto it3 = std::find(adjacency_list_.at(v1).begin(), adjacency_list_.at(v1).end(), v3);
        
        return it1 != adjacency_list_.at(v1).end() && 
               it2 != adjacency_list_.at(v2).end() && 
               it3 != adjacency_list_.at(v1).end();
    }
    
    void reorder_by_indices(const std::vector<size_type>& indices) {
        // Create temporary storage
        std::vector<core::Simplex> temp_simplices(simplices_.size());
        std::vector<double> temp_birth_times(birth_times_.size());
        
        // Reorder data
        for (size_type i = 0; i < indices.size(); ++i) {
            temp_simplices[i] = std::move(simplices_[indices[i]]);
            temp_birth_times[i] = birth_times_[indices[i]];
        }
        
        // Swap with temporary storage
        simplices_.swap(temp_simplices);
        birth_times_.swap(temp_birth_times);
    }
};

} // namespace tda::algorithms