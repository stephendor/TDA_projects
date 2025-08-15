#pragma once

#include "../core/types.hpp"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <algorithm>
#include <numeric>
#include <optional>

namespace tda::vector_stack {

/**
 * @brief Represents a persistence diagram with points (birth, death, dimension)
 * 
 * A persistence diagram is a collection of points in the plane where each point
 * represents a topological feature with its birth and death times. This class
 * provides functionality for manipulation, analysis, and serialization of
 * persistence diagrams.
 */
class PersistenceDiagram {
public:
    using Point = std::pair<double, double>; // (birth, death)
    using Dimension = int;
    using Index = size_t;
    
    /**
     * @brief Structure representing a single point in the persistence diagram
     */
    struct DiagramPoint {
        double birth;
        double death;
        int dimension;
        size_t birth_simplex;
        size_t death_simplex;
        
        DiagramPoint() : birth(0.0), death(0.0), dimension(0), birth_simplex(0), death_simplex(0) {}
        DiagramPoint(double b, double d, int dim, size_t b_sim = 0, size_t d_sim = 0)
            : birth(b), death(d), dimension(dim), birth_simplex(b_sim), death_simplex(d_sim) {}
        
        // Copy constructor and assignment
        DiagramPoint(const DiagramPoint&) = default;
        DiagramPoint& operator=(const DiagramPoint&) = default;
        
        // Move constructor and assignment
        DiagramPoint(DiagramPoint&&) = default;
        DiagramPoint& operator=(DiagramPoint&&) = default;
        
        /**
         * @brief Get the persistence (lifetime) of this feature
         * @return Death time minus birth time
         */
        double getPersistence() const noexcept {
            return death - birth;
        }
        
        /**
         * @brief Check if this feature has finite death time
         * @return True if death time is finite
         */
        bool isFinite() const noexcept {
            return std::isfinite(death);
        }
        
        /**
         * @brief Check if this feature has infinite death time
         * @return True if death time is infinite
         */
        bool isInfinite() const noexcept {
            return !std::isfinite(death);
        }
        
        /**
         * @brief Check if this feature is essential (infinite death)
         * @return True if the feature never dies
         */
        bool isEssential() const noexcept {
            return isInfinite();
        }
    };

    // Constructors
    PersistenceDiagram() = default;
    explicit PersistenceDiagram(const std::vector<DiagramPoint>& points);
    explicit PersistenceDiagram(const std::vector<tda::core::PersistencePair>& pairs);
    
    // Copy constructor and assignment
    PersistenceDiagram(const PersistenceDiagram&) = default;
    PersistenceDiagram& operator=(const PersistenceDiagram&) = default;
    
    // Move constructor and assignment
    PersistenceDiagram(PersistenceDiagram&&) = default;
    PersistenceDiagram& operator=(PersistenceDiagram&&) = default;
    
    // Destructor
    ~PersistenceDiagram() = default;
    
    // Core functionality
    void addPoint(const DiagramPoint& point);
    void addPoint(double birth, double death, int dimension, size_t birth_simplex = 0, size_t death_simplex = 0);
    void removePoint(size_t index);
    void clear();
    
    // Accessors
    const std::vector<DiagramPoint>& getPoints() const { return points_; }
    std::vector<DiagramPoint>& getPoints() { return points_; }
    size_t size() const { return points_.size(); }
    bool empty() const { return points_.empty(); }
    
    // Filtering and querying
    std::vector<DiagramPoint> getPointsByDimension(int dimension) const;
    std::vector<DiagramPoint> getFinitePoints() const;
    std::vector<DiagramPoint> getInfinitePoints() const;
    std::vector<DiagramPoint> getPointsInRange(double minBirth, double maxBirth, double minDeath, double maxDeath) const;
    
    // Statistics
    double getMinBirth() const;
    double getMaxBirth() const;
    double getMinDeath() const;
    double getMaxDeath() const;
    double getMinPersistence() const;
    double getMaxPersistence() const;
    double getAveragePersistence() const;
    
    // Betti numbers
    std::vector<size_t> getBettiNumbers() const;
    size_t getBettiNumber(int dimension) const;
    
    // Sorting and ordering
    void sortByBirth();
    void sortByDeath();
    void sortByPersistence();
    void sortByDimension();
    
    // Serialization
    std::string toCSV() const;
    std::string toTXT() const;
    
    // Static factory methods
    static PersistenceDiagram fromCSV(const std::string& csv);
    static PersistenceDiagram fromTXT(const std::string& txt);
    
    // Utility methods
    void normalize(double minValue = 0.0, double maxValue = 1.0);
    void filterByPersistence(double minPersistence);
    void filterByDimension(int minDimension, int maxDimension);
    
private:
    std::vector<DiagramPoint> points_;
    
    // Helper methods
    void validatePoint(const DiagramPoint& point) const;
    void updateStatistics();
};

/**
 * @brief Represents a persistence barcode with intervals (birth, death, dimension)
 * 
 * A persistence barcode is an alternative representation of persistent homology
 * where each topological feature is represented as an interval on the real line.
 * This class provides functionality for manipulation, analysis, and serialization
 * of persistence barcodes.
 */
class Barcode {
public:
    /**
     * @brief Structure representing a single interval in the persistence barcode
     */
    struct BarcodeInterval {
        double birth;
        double death;
        int dimension;
        size_t birth_simplex;
        size_t death_simplex;
        
        BarcodeInterval() : birth(0.0), death(0.0), dimension(0), birth_simplex(0), death_simplex(0) {}
        BarcodeInterval(double b, double d, int dim, size_t b_sim = 0, size_t d_sim = 0)
            : birth(b), death(d), dimension(dim), birth_simplex(b_sim), death_simplex(d_sim) {}
        
        // Copy constructor and assignment
        BarcodeInterval(const BarcodeInterval&) = default;
        BarcodeInterval& operator=(const BarcodeInterval&) = default;
        
        // Move constructor and assignment
        BarcodeInterval(BarcodeInterval&&) = default;
        BarcodeInterval& operator=(BarcodeInterval&&) = default;
        
        /**
         * @brief Get the length of this interval
         * @return Death time minus birth time
         */
        double getLength() const noexcept {
            return death - birth;
        }
        
        /**
         * @brief Check if this interval has finite death time
         * @return True if death time is finite
         */
        bool isFinite() const noexcept {
            return std::isfinite(death);
        }
        
        /**
         * @brief Check if this interval has infinite death time
         * @return True if death time is infinite
         */
        bool isInfinite() const noexcept {
            return !std::isfinite(death);
        }
        
        /**
         * @brief Check if this interval is essential (infinite death)
         * @return True if the interval never ends
         */
        bool isEssential() const noexcept {
            return isInfinite();
        }
    };

    // Constructors
    Barcode() = default;
    explicit Barcode(const std::vector<BarcodeInterval>& intervals);
    explicit Barcode(const std::vector<tda::core::PersistencePair>& pairs);
    explicit Barcode(const PersistenceDiagram& diagram);
    
    // Copy constructor and assignment
    Barcode(const Barcode&) = default;
    Barcode& operator=(const Barcode&) = default;
    
    // Move constructor and assignment
    Barcode(Barcode&&) = default;
    Barcode& operator=(Barcode&&) = default;
    
    // Destructor
    ~Barcode() = default;
    
    // Core functionality
    void addInterval(const BarcodeInterval& interval);
    void addInterval(double birth, double death, int dimension, size_t birth_simplex = 0, size_t death_simplex = 0);
    void removeInterval(size_t index);
    void clear();
    
    // Accessors
    const std::vector<BarcodeInterval>& getIntervals() const { return intervals_; }
    std::vector<BarcodeInterval>& getIntervals() { return intervals_; }
    size_t size() const { return intervals_.size(); }
    bool empty() const { return intervals_.empty(); }
    
    // Filtering and querying
    std::vector<BarcodeInterval> getIntervalsByDimension(int dimension) const;
    std::vector<BarcodeInterval> getFiniteIntervals() const;
    std::vector<BarcodeInterval> getInfiniteIntervals() const;
    std::vector<BarcodeInterval> getIntervalsInRange(double minBirth, double maxBirth, double minDeath, double maxDeath) const;
    
    // Statistics
    double getMinBirth() const;
    double getMaxBirth() const;
    double getMinDeath() const;
    double getMaxDeath() const;
    double getMinLength() const;
    double getMaxLength() const;
    double getAverageLength() const;
    
    // Betti numbers
    std::vector<size_t> getBettiNumbers() const;
    size_t getBettiNumber(int dimension) const;
    
    // Sorting and ordering
    void sortByBirth();
    void sortByDeath();
    void sortByLength();
    void sortByDimension();
    
    // Serialization
    std::string toCSV() const;
    std::string toTXT() const;
    
    // Static factory methods
    static Barcode fromCSV(const std::string& csv);
    static Barcode fromTXT(const std::string& txt);
    
    // Utility methods
    void normalize(double minValue = 0.0, double maxValue = 1.0);
    void filterByLength(double minLength);
    void filterByDimension(int minDimension, int maxDimension);
    
    // Conversion
    PersistenceDiagram toPersistenceDiagram() const;
    
private:
    std::vector<BarcodeInterval> intervals_;
    
    // Helper methods
    void validateInterval(const BarcodeInterval& interval) const;
    void updateStatistics();
};

// Utility functions for working with persistence data
namespace utils {
    
    /**
     * @brief Convert persistence pairs to a persistence diagram
     * @param pairs Vector of persistence pairs
     * @return PersistenceDiagram object
     */
    PersistenceDiagram pairsToDiagram(const std::vector<tda::core::PersistencePair>& pairs);
    
    /**
     * @brief Convert persistence pairs to a barcode
     * @param pairs Vector of persistence pairs
     * @return Barcode object
     */
    Barcode pairsToBarcode(const std::vector<tda::core::PersistencePair>& pairs);
    
    /**
     * @brief Convert a persistence diagram to a barcode
     * @param diagram PersistenceDiagram object
     * @return Barcode object
     */
    Barcode diagramToBarcode(const PersistenceDiagram& diagram);
    
    /**
     * @brief Convert a barcode to a persistence diagram
     * @param barcode Barcode object
     * @return PersistenceDiagram object
     */
    PersistenceDiagram barcodeToDiagram(const Barcode& barcode);
    
    /**
     * @brief Compute Wasserstein distance between two persistence diagrams
     * @param diagram1 First persistence diagram
     * @param diagram2 Second persistence diagram
     * @param p Order of the Wasserstein distance (default: 2)
     * @return Wasserstein distance value
     */
    double wassersteinDistance(const PersistenceDiagram& diagram1, const PersistenceDiagram& diagram2, double p = 2.0);
    
               /**
            * @brief Compute bottleneck distance between two persistence diagrams
            * @param diagram1 First persistence diagram
            * @param diagram2 Second persistence diagram
            * @return Bottleneck distance value
            */
           double bottleneckDistance(const PersistenceDiagram& diagram1, const PersistenceDiagram& diagram2);
           
           /**
            * @brief Compute Betti numbers at a specific filtration value
            * @param diagram The persistence diagram to analyze
            * @param epsilon The filtration value at which to compute Betti numbers
            * @return Vector of Betti numbers indexed by dimension
            */
           std::vector<size_t> computeBettiNumbersAtFiltration(const PersistenceDiagram& diagram, double epsilon);
           
           /**
            * @brief Compute Betti numbers at a specific filtration value for barcodes
            * @param barcode The persistence barcode to analyze
            * @param epsilon The filtration value at which to compute Betti numbers
            * @return Vector of Betti numbers indexed by dimension
            */
           std::vector<size_t> computeBettiNumbersAtFiltration(const Barcode& barcode, double epsilon);
           
           /**
            * @brief Compute Betti numbers at multiple filtration values
            * @param diagram The persistence diagram to analyze
            * @param epsilonValues Vector of filtration values to evaluate
            * @return Vector of Betti number vectors, one for each epsilon value
            */
           std::vector<std::vector<size_t>> computeBettiNumbersAtMultipleFiltrations(
               const PersistenceDiagram& diagram, 
               const std::vector<double>& epsilonValues);
           
           /**
            * @brief Compute Betti numbers at multiple filtration values for barcodes
            * @param barcode The persistence barcode to analyze
            * @param epsilonValues Vector of filtration values to evaluate
            * @return Vector of Betti number vectors, one for each epsilon value
            */
           std::vector<std::vector<size_t>> computeBettiNumbersAtMultipleFiltrations(
               const Barcode& barcode, 
               const std::vector<double>& epsilonValues);
           
           /**
            * @brief Compute Betti number curve (Betti numbers vs filtration value)
            * @param diagram The persistence diagram to analyze
            * @param numPoints Number of points to sample in the curve
            * @return Pair of vectors: (epsilon_values, betti_numbers_by_dimension)
            */
           std::pair<std::vector<double>, std::vector<std::vector<size_t>>> computeBettiNumberCurve(
               const PersistenceDiagram& diagram, 
               size_t numPoints = 100);
           
           /**
            * @brief Compute Betti number curve for barcodes
            * @param barcode The persistence barcode to analyze
            * @param numPoints Number of points to sample in the curve
            * @return Pair of vectors: (epsilon_values, betti_numbers_by_dimension)
            */
           std::pair<std::vector<double>, std::vector<std::vector<size_t>>> computeBettiNumberCurve(
               const Barcode& barcode, 
               size_t numPoints = 100);
           
           /**
            * @brief Find critical filtration values where Betti numbers change
            * @param diagram The persistence diagram to analyze
            * @return Vector of critical epsilon values
            */
           std::vector<double> findCriticalFiltrationValues(const PersistenceDiagram& diagram);
           
           /**
            * @brief Find critical filtration values for barcodes
            * @param barcode The persistence barcode to analyze
            * @return Vector of critical epsilon values
            */
           std::vector<double> findCriticalFiltrationValues(const Barcode& barcode);
    
} // namespace utils

} // namespace tda::vector_stack
