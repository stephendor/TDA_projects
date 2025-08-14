#include "../../../include/tda/core/types.hpp"
#include <algorithm>
#include <map>
#include <vector>
#include <unordered_map>

namespace tda::algorithms {

class SimpleVietorisRipsComplex {
private:
    std::vector<std::vector<core::Index>> simplices_;
    std::vector<double> birth_times_;
    std::unordered_map<core::Index, std::vector<core::Index>> adjacency_list_;
    double max_radius_;
    core::Dimension max_dimension_;
    
public:
    using size_type = std::size_t;
    
    SimpleVietorisRipsComplex(double max_radius, core::Dimension max_dimension = 3)
        : max_radius_(max_radius), max_dimension_(max_dimension) {}
    
    // Copy constructor
    SimpleVietorisRipsComplex(const SimpleVietorisRipsComplex& other)
        : simplices_(other.simplices_), birth_times_(other.birth_times_),
          adjacency_list_(other.adjacency_list_), max_radius_(other.max_radius_),
          max_dimension_(other.max_dimension_) {}
    
    // Move constructor
    SimpleVietorisRipsComplex(SimpleVietorisRipsComplex&& other) noexcept
        : simplices_(std::move(other.simplices_)), birth_times_(std::move(other.birth_times_)),
          adjacency_list_(std::move(other.adjacency_list_)), max_radius_(other.max_radius_),
          max_dimension_(other.max_dimension_) {}
    
    // Assignment operators
    SimpleVietorisRipsComplex& operator=(const SimpleVietorisRipsComplex& other) {
        if (this != &other) {
            simplices_ = other.simplices_;
            birth_times_ = other.birth_times_;
            adjacency_list_ = other.adjacency_list_;
            max_radius_ = other.max_radius_;
            max_dimension_ = other.max_dimension_;
        }
        return *this;
    }
    
    SimpleVietorisRipsComplex& operator=(SimpleVietorisRipsComplex&& other) noexcept {
        if (this != &other) {
            simplices_ = std::move(other.simplices_);
            birth_times_ = std::move(other.birth_times_);
            adjacency_list_ = std::move(other.adjacency_list_);
            max_radius_ = other.max_radius_;
            max_dimension_ = other.max_dimension_;
        }
        return *this;
    }
    
    // Configuration
    void set_max_radius(double radius) { max_radius_ = radius; }
    void set_max_dimension(core::Dimension dim) { max_dimension_ = dim; }
    
    double max_radius() const { return max_radius_; }
    core::Dimension max_dimension() const { return max_dimension_; }
    
    // Data access
    const std::vector<std::vector<core::Index>>& simplices() const { return simplices_; }
    const std::vector<double>& birth_times() const { return birth_times_; }
    const std::unordered_map<core::Index, std::vector<core::Index>>& adjacency_list() const { return adjacency_list_; }
    
    // Capacity
    size_type size() const { return simplices_.size(); }
    bool empty() const { return simplices_.empty(); }
    
    // Complex construction from point cloud
    void build_from_point_cloud(const std::vector<std::vector<double>>& points) {
        simplices_.clear();
        birth_times_.clear();
        adjacency_list_.clear();
        
        size_t n = points.size();
        if (n == 0) return;
        
        // Add vertices (0-simplices)
        for (size_t i = 0; i < n; ++i) {
            simplices_.emplace_back(std::vector<core::Index>{static_cast<core::Index>(i)});
            birth_times_.push_back(0.0);
        }
        
        // Build edges (1-simplices) and adjacency list
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double distance = compute_distance(points[i], points[j]);
                if (distance <= max_radius_) {
                    simplices_.emplace_back(std::vector<core::Index>{
                        static_cast<core::Index>(i), static_cast<core::Index>(j)
                    });
                    birth_times_.push_back(distance);
                    
                    // Update adjacency list
                    adjacency_list_[static_cast<core::Index>(i)].push_back(static_cast<core::Index>(j));
                    adjacency_list_[static_cast<core::Index>(j)].push_back(static_cast<core::Index>(i));
                }
            }
        }
        
        // Build higher-dimensional simplices if requested
        if (max_dimension_ >= 2) {
            build_triangles();
        }
        if (max_dimension_ >= 3) {
            build_tetrahedra();
        }
    }
    
    // Complex construction from distance matrix
    void build_from_distance_matrix(const std::vector<std::vector<double>>& distances) {
        simplices_.clear();
        birth_times_.clear();
        adjacency_list_.clear();
        
        size_t n = distances.size();
        if (n == 0) return;
        
        // Add vertices
        for (size_t i = 0; i < n; ++i) {
            simplices_.emplace_back(std::vector<core::Index>{static_cast<core::Index>(i)});
            birth_times_.push_back(0.0);
        }
        
        // Build edges
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double distance = distances[i][j];
                if (distance <= max_radius_) {
                    simplices_.emplace_back(std::vector<core::Index>{
                        static_cast<core::Index>(i), static_cast<core::Index>(j)
                    });
                    birth_times_.push_back(distance);
                    
                    adjacency_list_[static_cast<core::Index>(i)].push_back(static_cast<core::Index>(j));
                    adjacency_list_[static_cast<core::Index>(j)].push_back(static_cast<core::Index>(i));
                }
            }
        }
        
        // Build higher-dimensional simplices
        if (max_dimension_ >= 2) {
            build_triangles();
        }
        if (max_dimension_ >= 3) {
            build_tetrahedra();
        }
    }
    
    // Filtration operations
    void sort_by_birth_time() {
        std::vector<size_type> indices(simplices_.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), [this](size_type a, size_type b) {
            return birth_times_[a] < birth_times_[b];
        });
        
        reorder_by_indices(indices);
    }
    
    void sort_by_dimension() {
        std::vector<size_type> indices(simplices_.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), [this](size_type a, size_type b) {
            return simplices_[a].size() < simplices_[b].size();
        });
        
        reorder_by_indices(indices);
    }
    
    // Query operations
    std::vector<size_type> find_simplices_by_dimension(core::Dimension dim) const {
        std::vector<size_type> result;
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (static_cast<core::Dimension>(simplices_[i].size() - 1) == dim) {
                result.push_back(i);
            }
        }
        return result;
    }
    
    std::vector<size_type> find_simplices_in_time_range(double min_time, double max_time) const {
        std::vector<size_type> result;
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (birth_times_[i] >= min_time && birth_times_[i] <= max_time) {
                result.push_back(i);
            }
        }
        return result;
    }
    
    // Utility methods
    core::Dimension get_simplex_dimension(size_type index) const {
        if (index >= simplices_.size()) return 0;
        return static_cast<core::Dimension>(simplices_[index].size() - 1);
    }
    
    double get_simplex_birth_time(size_type index) const {
        if (index >= birth_times_.size()) return 0.0;
        return birth_times_[index];
    }
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(simplices_, birth_times_, adjacency_list_, max_radius_, max_dimension_);
    }
    
private:
    // Helper method to compute distance between two points
    double compute_distance(const std::vector<double>& p1, const std::vector<double>& p2) const {
        if (p1.size() != p2.size()) return std::numeric_limits<double>::infinity();
        
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    // Helper method to get edge birth time
    double get_edge_birth_time(core::Index v1, core::Index v2) const {
        // Find the edge in the simplices list
        for (size_type i = 0; i < simplices_.size(); ++i) {
            const auto& simplex = simplices_[i];
            if (simplex.size() == 2) {
                if ((simplex[0] == v1 && simplex[1] == v2) ||
                    (simplex[0] == v2 && simplex[1] == v1)) {
                    return birth_times_[i];
                }
            }
        }
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
                    auto it = std::find(adjacency_list_.at(neighbor1).begin(), 
                                       adjacency_list_.at(neighbor1).end(), neighbor2);
                    if (it != adjacency_list_.at(neighbor1).end()) {
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
        std::sort(triangles.begin(), triangles.end());
        
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
        std::sort(tetrahedra.begin(), tetrahedra.end());
        
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
        std::vector<std::vector<core::Index>> temp_simplices(simplices_.size());
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