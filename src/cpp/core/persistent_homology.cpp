#include "../../../include/tda/core/types.hpp"
#include <algorithm>
#include <execution>
#include <queue>
#include <unordered_map>
#include <vector>

namespace tda::core {

class PersistentHomology {
private:
    Filtration filtration_;
    std::vector<PersistencePair> persistence_pairs_;
    std::vector<BettiNumbers> betti_numbers_;
    bool is_computed_;
    
public:
    // Type aliases
    using size_type = std::size_t;
    using iterator = std::vector<PersistencePair>::iterator;
    using const_iterator = std::vector<PersistencePair>::const_iterator;
    
    // Constructors
    PersistentHomology() = default;
    
    explicit PersistentHomology(const Filtration& filtration)
        : filtration_(filtration), is_computed_(false) {}
    
    PersistentHomology(Filtration&& filtration)
        : filtration_(std::move(filtration)), is_computed_(false) {}
    
    // Copy constructor
    PersistentHomology(const PersistentHomology& other)
        : filtration_(other.filtration_), persistence_pairs_(other.persistence_pairs_),
          betti_numbers_(other.betti_numbers_), is_computed_(other.is_computed_) {}
    
    // Move constructor
    PersistentHomology(PersistentHomology&& other) noexcept
        : filtration_(std::move(other.filtration_)), persistence_pairs_(std::move(other.persistence_pairs_)),
          betti_numbers_(std::move(other.betti_numbers_)), is_computed_(other.is_computed_) {
        other.is_computed_ = false;
    }
    
    // Destructor
    ~PersistentHomology() = default;
    
    // Assignment operators
    PersistentHomology& operator=(const PersistentHomology& other) {
        if (this != &other) {
            filtration_ = other.filtration_;
            persistence_pairs_ = other.persistence_pairs_;
            betti_numbers_ = other.betti_numbers_;
            is_computed_ = other.is_computed_;
        }
        return *this;
    }
    
    PersistentHomology& operator=(PersistentHomology&& other) noexcept {
        if (this != &other) {
            filtration_ = std::move(other.filtration_);
            persistence_pairs_ = std::move(other.persistence_pairs_);
            betti_numbers_ = std::move(other.betti_numbers_);
            is_computed_ = other.is_computed_;
            other.is_computed_ = false;
        }
        return *this;
    }
    
    // Data access
    const Filtration& filtration() const noexcept { return filtration_; }
    Filtration& filtration() noexcept { return filtration_; }
    
    const std::vector<PersistencePair>& persistence_pairs() const noexcept { return persistence_pairs_; }
    std::vector<PersistencePair>& persistence_pairs() noexcept { return persistence_pairs_; }
    
    const std::vector<BettiNumbers>& betti_numbers() const noexcept { return betti_numbers_; }
    std::vector<BettiNumbers>& betti_numbers() noexcept { return betti_numbers_; }
    
    // Iterators
    iterator begin() noexcept { return persistence_pairs_.begin(); }
    const_iterator begin() const noexcept { return persistence_pairs_.begin(); }
    const_iterator cbegin() const noexcept { return persistence_pairs_.cbegin(); }
    
    iterator end() noexcept { return persistence_pairs_.end(); }
    const_iterator end() const noexcept { return persistence_pairs_.end(); }
    const_iterator cend() const noexcept { return persistence_pairs_.cend(); }
    
    // Capacity
    size_type size() const noexcept { return persistence_pairs_.size(); }
    bool empty() const noexcept { return persistence_pairs_.empty(); }
    
    // Computation
    void compute() {
        if (is_computed_) {
            return;
        }
        
        // Ensure filtration is sorted by birth time
        filtration_.sort_by_birth_time();
        
        // Clear previous results
        persistence_pairs_.clear();
        betti_numbers_.clear();
        
        // Compute persistent homology using matrix reduction
        compute_matrix_reduction();
        
        // Compute Betti numbers
        compute_betti_numbers();
        
        is_computed_ = true;
    }
    
    void recompute() {
        is_computed_ = false;
        compute();
    }
    
    bool is_computed() const noexcept { return is_computed_; }
    
    // Query operations
    std::vector<PersistencePair> get_pairs_by_dimension(Dimension dim) const {
        if (!is_computed_) {
            return {};
        }
        
        std::vector<PersistencePair> pairs;
        pairs.reserve(persistence_pairs_.size());
        
        for (const auto& pair : persistence_pairs_) {
            if (pair.dimension == dim) {
                pairs.push_back(pair);
            }
        }
        
        return pairs;
    }
    
    std::vector<PersistencePair> get_pairs_by_persistence_range(double min_pers, double max_pers) const {
        if (!is_computed_) {
            return {};
        }
        
        std::vector<PersistencePair> pairs;
        pairs.reserve(persistence_pairs_.size());
        
        for (const auto& pair : persistence_pairs_) {
            double persistence = pair.get_persistence();
            if (persistence >= min_pers && persistence <= max_pers) {
                pairs.push_back(pair);
            }
        }
        
        return pairs;
    }
    
    std::vector<PersistencePair> get_finite_pairs() const {
        if (!is_computed_) {
            return {};
        }
        
        std::vector<PersistencePair> pairs;
        pairs.reserve(persistence_pairs_.size());
        
        for (const auto& pair : persistence_pairs_) {
            if (pair.is_finite()) {
                pairs.push_back(pair);
            }
        }
        
        return pairs;
    }
    
    std::vector<PersistencePair> get_infinite_pairs() const {
        if (!is_computed_) {
            return {};
        }
        
        std::vector<PersistencePair> pairs;
        pairs.reserve(persistence_pairs_.size());
        
        for (const auto& pair : persistence_pairs_) {
            if (pair.is_infinite()) {
                pairs.push_back(pair);
            }
        }
        
        return pairs;
    }
    
    // Statistical operations
    double mean_persistence() const {
        if (!is_computed_ || persistence_pairs_.empty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        size_type count = 0;
        
        for (const auto& pair : persistence_pairs_) {
            if (pair.is_finite()) {
                sum += pair.get_persistence();
                ++count;
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    double median_persistence() const {
        if (!is_computed_ || persistence_pairs_.empty()) {
            return 0.0;
        }
        
        std::vector<double> persistences;
        persistences.reserve(persistence_pairs_.size());
        
        for (const auto& pair : persistence_pairs_) {
            if (pair.is_finite()) {
                persistences.push_back(pair.get_persistence());
            }
        }
        
        if (persistences.empty()) {
            return 0.0;
        }
        
        std::sort(persistences.begin(), persistences.end());
        size_type n = persistences.size();
        
        if (n % 2 == 0) {
            return (persistences[n/2 - 1] + persistences[n/2]) / 2.0;
        } else {
            return persistences[n/2];
        }
    }
    
    double persistence_variance() const {
        if (!is_computed_ || persistence_pairs_.size() < 2) {
            return 0.0;
        }
        
        double mean = mean_persistence();
        double sum_sq = 0.0;
        size_type count = 0;
        
        for (const auto& pair : persistence_pairs_) {
            if (pair.is_finite()) {
                double diff = pair.get_persistence() - mean;
                sum_sq += diff * diff;
                ++count;
            }
        }
        
        return count > 1 ? sum_sq / (count - 1) : 0.0;
    }
    
    // Utility methods
    void clear() noexcept {
        persistence_pairs_.clear();
        betti_numbers_.clear();
        is_computed_ = false;
    }
    
    void reserve(size_type capacity) {
        persistence_pairs_.reserve(capacity);
        betti_numbers_.reserve(capacity);
    }
    
    void shrink_to_fit() {
        persistence_pairs_.shrink_to_fit();
        betti_numbers_.shrink_to_fit();
    }
    
    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(filtration_, persistence_pairs_, betti_numbers_, is_computed_);
    }
    
private:
    void compute_matrix_reduction() {
        // This is a simplified matrix reduction algorithm
        // In practice, you would use more sophisticated algorithms like
        // the standard algorithm or the clearing algorithm
        
        size_type n = filtration_.size();
        if (n == 0) return;
        
        // Create boundary matrix
        std::vector<std::vector<bool>> boundary_matrix(n, std::vector<bool>(n, false));
        
        // Fill boundary matrix
        for (size_type i = 0; i < n; ++i) {
            const auto& simplex = filtration_[i];
            auto faces = simplex.get_faces();
            
            for (const auto& face : faces) {
                // Find face index in filtration
                for (size_type j = 0; j < n; ++j) {
                    if (filtration_[j] == face) {
                        boundary_matrix[i][j] = true;
                        break;
                    }
                }
            }
        }
        
        // Simple matrix reduction (Gaussian elimination)
        std::vector<Index> low = std::vector<Index>(n, n);
        
        for (size_type j = 0; j < n; ++j) {
            // Find lowest 1 in column j
            Index low_j = n;
            for (size_type i = 0; i < n; ++i) {
                if (boundary_matrix[i][j]) {
                    low_j = i;
                    break;
                }
            }
            
            if (low_j < n) {
                low[j] = low_j;
                
                // Eliminate other 1s in column j
                for (size_type k = j + 1; k < n; ++k) {
                    if (boundary_matrix[low_j][k]) {
                        // Add column k to column j
                        for (size_type i = 0; i < n; ++i) {
                            boundary_matrix[i][j] = boundary_matrix[i][j] ^ boundary_matrix[i][k];
                        }
                    }
                }
            }
        }
        
        // Extract persistence pairs
        for (size_type j = 0; j < n; ++j) {
            if (low[j] < n) {
                // Finite pair
                double birth_time = filtration_.get_birth_time(j);
                double death_time = filtration_.get_birth_time(low[j]);
                Dimension dim = filtration_[j].dimension();
                
                persistence_pairs_.emplace_back(birth_time, death_time, dim, j, low[j]);
            } else {
                // Infinite pair (birth simplex)
                double birth_time = filtration_.get_birth_time(j);
                Dimension dim = filtration_[j].dimension();
                
                persistence_pairs_.emplace_back(birth_time, std::numeric_limits<double>::infinity(), 
                                             dim, j, j);
            }
        }
    }
    
    void compute_betti_numbers() {
        if (persistence_pairs_.empty()) {
            return;
        }
        
        // Find maximum dimension
        Dimension max_dim = 0;
        for (const auto& pair : persistence_pairs_) {
            max_dim = std::max(max_dim, pair.dimension);
        }
        
        // Initialize Betti numbers
        betti_numbers_.resize(max_dim + 1);
        for (auto& betti : betti_numbers_) {
            betti = BettiNumbers(max_dim);
        }
        
        // Count infinite pairs (births) for each dimension
        for (const auto& pair : persistence_pairs_) {
            if (pair.is_infinite()) {
                betti_numbers_[pair.dimension][pair.dimension]++;
            }
        }
    }
};

} // namespace tda::core