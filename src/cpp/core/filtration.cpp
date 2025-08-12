#include "../../../include/tda/core/types.hpp"
#include <algorithm>
#include <execution>
#include <stdexcept>
#include <unordered_map>

namespace tda::core {

class Filtration {
private:
    std::vector<Simplex> simplices_;
    std::unordered_map<Index, Index> simplex_to_index_;
    std::vector<double> birth_times_;
    std::vector<double> death_times_;
    bool is_sorted_;
    
public:
    // Type aliases
    using size_type = std::size_t;
    using iterator = std::vector<Simplex>::iterator;
    using const_iterator = std::vector<Simplex>::const_iterator;
    
    // Constructors
    Filtration() = default;
    
    explicit Filtration(size_type capacity) {
        simplices_.reserve(capacity);
        birth_times_.reserve(capacity);
        death_times_.reserve(capacity);
        is_sorted_ = true;
    }
    
    // Copy constructor
    Filtration(const Filtration& other)
        : simplices_(other.simplices_), simplex_to_index_(other.simplex_to_index_),
          birth_times_(other.birth_times_), death_times_(other.death_times_),
          is_sorted_(other.is_sorted_) {}
    
    // Move constructor
    Filtration(Filtration&& other) noexcept
        : simplices_(std::move(other.simplices_)), simplex_to_index_(std::move(other.simplex_to_index_)),
          birth_times_(std::move(other.birth_times_)), death_times_(std::move(other.death_times_)),
          is_sorted_(other.is_sorted_) {
        other.is_sorted_ = true;
    }
    
    // Destructor
    ~Filtration() = default;
    
    // Assignment operators
    Filtration& operator=(const Filtration& other) {
        if (this != &other) {
            simplices_ = other.simplices_;
            simplex_to_index_ = other.simplex_to_index_;
            birth_times_ = other.birth_times_;
            death_times_ = other.death_times_;
            is_sorted_ = other.is_sorted_;
        }
        return *this;
    }
    
    Filtration& operator=(Filtration&& other) noexcept {
        if (this != &other) {
            simplices_ = std::move(other.simplices_);
            simplex_to_index_ = std::move(other.simplex_to_index_);
            birth_times_ = std::move(other.birth_times_);
            death_times_ = std::move(other.death_times_);
            is_sorted_ = other.is_sorted_;
            other.is_sorted_ = true;
        }
        return *this;
    }
    
    // Element access
    Simplex& operator[](size_type index) {
        return simplices_[index];
    }
    
    const Simplex& operator[](size_type index) const {
        return simplices_[index];
    }
    
    Simplex& at(size_type index) {
        if (index >= simplices_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        return simplices_[index];
    }
    
    const Simplex& at(size_type index) const {
        if (index >= simplices_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        return simplices_[index];
    }
    
    // Capacity
    size_type size() const noexcept { return simplices_.size(); }
    bool empty() const noexcept { return simplices_.empty(); }
    
    // Iterators
    iterator begin() noexcept { return simplices_.begin(); }
    const_iterator begin() const noexcept { return simplices_.begin(); }
    const_iterator cbegin() const noexcept { return simplices_.cbegin(); }
    
    iterator end() noexcept { return simplices_.end(); }
    const_iterator end() const noexcept { return simplices_.end(); }
    const_iterator cend() const noexcept { return simplices_.cend(); }
    
    // Data access
    std::vector<Simplex>& simplices() noexcept { return simplices_; }
    const std::vector<Simplex>& simplices() const noexcept { return simplices_; }
    
    std::vector<double>& birth_times() noexcept { return birth_times_; }
    const std::vector<double>& birth_times() const noexcept { return birth_times_; }
    
    std::vector<double>& death_times() noexcept { return death_times_; }
    const std::vector<double>& death_times() const noexcept { return death_times_; }
    
    // Simplex management
    void add_simplex(const Simplex& simplex, double birth_time = 0.0) {
        Index index = static_cast<Index>(simplices_.size());
        simplices_.push_back(simplex);
        birth_times_.push_back(birth_time);
        death_times_.push_back(std::numeric_limits<double>::infinity());
        
        // Store mapping for quick lookup
        simplex_to_index_[index] = index;
        is_sorted_ = false;
    }
    
    void add_simplex(Simplex&& simplex, double birth_time = 0.0) {
        Index index = static_cast<Index>(simplices_.size());
        simplices_.push_back(std::move(simplex));
        birth_times_.push_back(birth_time);
        death_times_.push_back(std::numeric_limits<double>::infinity());
        
        // Store mapping for quick lookup
        simplex_to_index_[index] = index;
        is_sorted_ = false;
    }
    
    void remove_simplex(size_type index) {
        if (index >= simplices_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        
        simplices_.erase(simplices_.begin() + index);
        birth_times_.erase(birth_times_.begin() + index);
        death_times_.erase(death_times_.begin() + index);
        
        // Rebuild mapping
        rebuild_mapping();
        is_sorted_ = false;
    }
    
    void clear() noexcept {
        simplices_.clear();
        birth_times_.clear();
        death_times_.clear();
        simplex_to_index_.clear();
        is_sorted_ = true;
    }
    
    // Time management
    void set_birth_time(size_type index, double time) {
        if (index >= birth_times_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        birth_times_[index] = time;
        is_sorted_ = false;
    }
    
    void set_death_time(size_type index, double time) {
        if (index >= death_times_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        death_times_[index] = time;
    }
    
    double get_birth_time(size_type index) const {
        if (index >= birth_times_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        return birth_times_[index];
    }
    
    double get_death_time(size_type index) const {
        if (index >= death_times_.size()) {
            throw std::out_of_range("Filtration index out of range");
        }
        return death_times_[index];
    }
    
    // Filtration ordering
    void sort_by_birth_time() {
        if (is_sorted_) {
            return;
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
        is_sorted_ = true;
    }
    
    void sort_by_dimension() {
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
        is_sorted_ = false; // Not sorted by birth time anymore
    }
    
    void sort_by_persistence() {
        // Create index vector for sorting
        std::vector<size_type> indices(simplices_.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort indices by persistence (descending)
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(),
                  [this](size_type i, size_type j) {
                      double pers_i = death_times_[i] - birth_times_[i];
                      double pers_j = death_times_[j] - birth_times_[j];
                      
                      if (std::isinf(pers_i) && std::isinf(pers_j)) {
                          return birth_times_[i] < birth_times_[j];
                      }
                      if (std::isinf(pers_i)) return false;
                      if (std::isinf(pers_j)) return true;
                      
                      return pers_i > pers_j;
                  });
        
        // Reorder data according to sorted indices
        reorder_by_indices(indices);
        is_sorted_ = false;
    }
    
    // Query operations
    std::vector<size_type> find_simplices_by_dimension(Dimension dim) const {
        std::vector<size_type> indices;
        indices.reserve(simplices_.size());
        
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (simplices_[i].dimension() == dim) {
                indices.push_back(i);
            }
        }
        
        return indices;
    }
    
    std::vector<size_type> find_simplices_in_time_range(double start_time, double end_time) const {
        std::vector<size_type> indices;
        indices.reserve(simplices_.size());
        
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (birth_times_[i] >= start_time && birth_times_[i] <= end_time) {
                indices.push_back(i);
            }
        }
        
        return indices;
    }
    
    std::vector<size_type> find_finite_simplices() const {
        std::vector<size_type> indices;
        indices.reserve(simplices_.size());
        
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (!std::isinf(death_times_[i])) {
                indices.push_back(i);
            }
        }
        
        return indices;
    }
    
    std::vector<size_type> find_infinite_simplices() const {
        std::vector<size_type> indices;
        indices.reserve(simplices_.size());
        
        for (size_type i = 0; i < simplices_.size(); ++i) {
            if (std::isinf(death_times_[i])) {
                indices.push_back(i);
            }
        }
        
        return indices;
    }
    
    // Utility methods
    bool is_sorted() const noexcept { return is_sorted_; }
    
    void reserve(size_type capacity) {
        simplices_.reserve(capacity);
        birth_times_.reserve(capacity);
        death_times_.reserve(capacity);
    }
    
    void shrink_to_fit() {
        simplices_.shrink_to_fit();
        birth_times_.shrink_to_fit();
        death_times_.shrink_to_fit();
    }
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(simplices_, simplex_to_index_, birth_times_, death_times_, is_sorted_);
    }
    
private:
    void reorder_by_indices(const std::vector<size_type>& indices) {
        // Create temporary storage
        std::vector<Simplex> temp_simplices(simplices_.size());
        std::vector<double> temp_birth_times(birth_times_.size());
        std::vector<double> temp_death_times(death_times_.size());
        
        // Reorder data
        for (size_type i = 0; i < indices.size(); ++i) {
            temp_simplices[i] = std::move(simplices_[indices[i]]);
            temp_birth_times[i] = birth_times_[indices[i]];
            temp_death_times[i] = death_times_[indices[i]];
        }
        
        // Swap with temporary storage
        simplices_.swap(temp_simplices);
        birth_times_.swap(temp_birth_times);
        death_times_.swap(temp_death_times);
        
        // Rebuild mapping
        rebuild_mapping();
    }
    
    void rebuild_mapping() {
        simplex_to_index_.clear();
        for (size_type i = 0; i < simplices_.size(); ++i) {
            simplex_to_index_[i] = i;
        }
    }
};

} // namespace tda::core