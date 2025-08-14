#include "../../../include/tda/core/types.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace tda::core {

class Simplex {
private:
    std::vector<Index> vertices_;
    Dimension dimension_;
    double birth_time_;
    double death_time_;
    Index birth_simplex_;
    Index death_simplex_;
    
public:
    // Type aliases
    using vertex_type = Index;
    using size_type = std::size_t;
    using iterator = std::vector<Index>::iterator;
    using const_iterator = std::vector<Index>::const_iterator;
    
    // Constructors
    Simplex() = default;
    
    explicit Simplex(const std::vector<Index>& vertices)
        : vertices_(vertices), dimension_(vertices.empty() ? 0 : static_cast<Dimension>(vertices.size() - 1)) {
        std::sort(vertices_.begin(), vertices_.end());
        birth_time_ = 0.0;
        death_time_ = std::numeric_limits<double>::infinity();
        birth_simplex_ = 0;
        death_simplex_ = 0;
    }
    
    Simplex(std::vector<Index>&& vertices)
        : vertices_(std::move(vertices)), dimension_(vertices_.empty() ? 0 : static_cast<Dimension>(vertices_.size() - 1)) {
        std::sort(vertices_.begin(), vertices_.end());
        birth_time_ = 0.0;
        death_time_ = std::numeric_limits<double>::infinity();
        birth_simplex_ = 0;
        death_simplex_ = 0;
    }
    
    Simplex(std::initializer_list<Index> vertices)
        : vertices_(vertices), dimension_(vertices.size() == 0 ? 0 : static_cast<Dimension>(vertices.size() - 1)) {
        std::sort(vertices_.begin(), vertices_.end());
        birth_time_ = 0.0;
        death_time_ = std::numeric_limits<double>::infinity();
        birth_simplex_ = 0;
        death_simplex_ = 0;
    }
    
    // Copy constructor
    Simplex(const Simplex& other)
        : vertices_(other.vertices_), dimension_(other.dimension_),
          birth_time_(other.birth_time_), death_time_(other.death_time_),
          birth_simplex_(other.birth_simplex_), death_simplex_(other.death_simplex_) {}
    
    // Move constructor
    Simplex(Simplex&& other) noexcept
        : vertices_(std::move(other.vertices_)), dimension_(other.dimension_),
          birth_time_(other.birth_time_), death_time_(other.death_time_),
          birth_simplex_(other.birth_simplex_), death_simplex_(other.death_simplex_) {
        other.dimension_ = 0;
        other.birth_time_ = 0.0;
        other.death_time_ = 0.0;
        other.birth_simplex_ = 0;
        other.death_simplex_ = 0;
    }
    
    // Destructor
    ~Simplex() = default;
    
    // Assignment operators
    Simplex& operator=(const Simplex& other) {
        if (this != &other) {
            vertices_ = other.vertices_;
            dimension_ = other.dimension_;
            birth_time_ = other.birth_time_;
            death_time_ = other.death_time_;
            birth_simplex_ = other.birth_simplex_;
            death_simplex_ = other.death_simplex_;
        }
        return *this;
    }
    
    Simplex& operator=(Simplex&& other) noexcept {
        if (this != &other) {
            vertices_ = std::move(other.vertices_);
            dimension_ = other.dimension_;
            birth_time_ = other.birth_time_;
            death_time_ = other.death_time_;
            birth_simplex_ = other.birth_simplex_;
            death_simplex_ = other.death_simplex_;
            other.dimension_ = 0;
            other.birth_time_ = 0.0;
            other.death_time_ = 0.0;
            other.birth_simplex_ = 0;
            other.death_simplex_ = 0;
        }
        return *this;
    }
    
    // Element access
    vertex_type& operator[](size_type index) {
        return vertices_[index];
    }
    
    const vertex_type& operator[](size_type index) const {
        return vertices_[index];
    }
    
    vertex_type& at(size_type index) {
        if (index >= vertices_.size()) {
            throw std::out_of_range("Simplex index out of range");
        }
        return vertices_[index];
    }
    
    const vertex_type& at(size_type index) const {
        if (index >= vertices_.size()) {
            throw std::out_of_range("Simplex index out of range");
        }
        return vertices_[index];
    }
    
    // Capacity
    size_type size() const noexcept { return vertices_.size(); }
    Dimension dimension() const noexcept { return dimension_; }
    bool empty() const noexcept { return vertices_.empty(); }
    
    // Iterators
    iterator begin() noexcept { return vertices_.begin(); }
    const_iterator begin() const noexcept { return vertices_.begin(); }
    const_iterator cbegin() const noexcept { return vertices_.cbegin(); }
    
    iterator end() noexcept { return vertices_.end(); }
    const_iterator end() const noexcept { return vertices_.end(); }
    const_iterator cend() const noexcept { return vertices_.cend(); }
    
    // Data access
    std::vector<Index>& vertices() noexcept { return vertices_; }
    const std::vector<Index>& vertices() const noexcept { return vertices_; }
    
    // Vertex management
    void add_vertex(Index vertex) {
        vertices_.push_back(vertex);
        std::sort(vertices_.begin(), vertices_.end());
        dimension_ = static_cast<Dimension>(vertices_.size() - 1);
    }
    
    void remove_vertex(Index vertex) {
        auto it = std::find(vertices_.begin(), vertices_.end(), vertex);
        if (it != vertices_.end()) {
            vertices_.erase(it);
            dimension_ = static_cast<Dimension>(vertices_.size() - 1);
        }
    }
    
    bool contains_vertex(Index vertex) const {
        return std::binary_search(vertices_.begin(), vertices_.end(), vertex);
    }
    
    // Face operations
    std::vector<Simplex> get_faces() const {
        if (dimension_ == 0) {
            return {};
        }
        
        std::vector<Simplex> faces;
        faces.reserve(vertices_.size());
        
        for (size_type i = 0; i < vertices_.size(); ++i) {
            std::vector<Index> face_vertices;
            face_vertices.reserve(vertices_.size() - 1);
            
            for (size_type j = 0; j < vertices_.size(); ++j) {
                if (i != j) {
                    face_vertices.push_back(vertices_[j]);
                }
            }
            
            faces.emplace_back(std::move(face_vertices));
        }
        
        return faces;
    }
    
    std::vector<Simplex> get_cofaces() const {
        // This would typically be implemented in the complex structure
        // For now, return empty vector
        return {};
    }
    
    // Boundary operations
    std::vector<Simplex> get_boundary() const {
        return get_faces();
    }
    
    // Persistence operations
    void set_birth_time(double time) noexcept { birth_time_ = time; }
    void set_death_time(double time) noexcept { death_time_ = time; }
    void set_birth_simplex(Index simplex) noexcept { birth_simplex_ = simplex; }
    void set_death_simplex(Index simplex) noexcept { death_simplex_ = simplex; }
    
    double birth_time() const noexcept { return birth_time_; }
    double death_time() const noexcept { return death_time_; }
    Index birth_simplex() const noexcept { return birth_simplex_; }
    Index death_simplex() const noexcept { return death_simplex_; }
    
    double persistence() const noexcept {
        if (std::isinf(death_time_)) {
            return std::numeric_limits<double>::infinity();
        }
        return death_time_ - birth_time_;
    }
    
    bool is_finite() const noexcept {
        return !std::isinf(death_time_);
    }
    
    bool is_infinite() const noexcept {
        return std::isinf(death_time_);
    }
    
    // Geometric operations
    bool is_face_of(const Simplex& other) const {
        if (dimension_ >= other.dimension_) {
            return false;
        }
        
        // Check if all vertices of this simplex are in the other
        for (Index vertex : vertices_) {
            if (!other.contains_vertex(vertex)) {
                return false;
            }
        }
        return true;
    }
    
    bool is_coface_of(const Simplex& other) const {
        return other.is_face_of(*this);
    }
    
    // Comparison operators
    bool operator==(const Simplex& other) const {
        return vertices_ == other.vertices_;
    }
    
    bool operator!=(const Simplex& other) const {
        return vertices_ != other.vertices_;
    }
    
    bool operator<(const Simplex& other) const {
        if (dimension_ != other.dimension_) {
            return dimension_ < other.dimension_;
        }
        return vertices_ < other.vertices_;
    }
    
    bool operator<=(const Simplex& other) const {
        return (*this < other) || (*this == other);
    }
    
    bool operator>(const Simplex& other) const {
        return !(*this <= other);
    }
    
    bool operator>=(const Simplex& other) const {
        return !(*this < other);
    }
    
    // Utility methods
    void clear() noexcept {
        vertices_.clear();
        dimension_ = 0;
        birth_time_ = 0.0;
        death_time_ = std::numeric_limits<double>::infinity();
        birth_simplex_ = 0;
        death_simplex_ = 0;
    }
    
    void reserve(size_type capacity) {
        vertices_.reserve(capacity);
    }
    
    void shrink_to_fit() {
        vertices_.shrink_to_fit();
    }
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(vertices_, dimension_, birth_time_, death_time_, birth_simplex_, death_simplex_);
    }
    
    // Static factory methods
    static Simplex vertex(Index v) {
        return Simplex{v};
    }
    
    static Simplex edge(Index v1, Index v2) {
        return Simplex{v1, v2};
    }
    
    static Simplex triangle(Index v1, Index v2, Index v3) {
        return Simplex{v1, v2, v3};
    }
    
    static Simplex tetrahedron(Index v1, Index v2, Index v3, Index v4) {
        return Simplex{v1, v2, v3, v4};
    }
};

} // namespace tda::core