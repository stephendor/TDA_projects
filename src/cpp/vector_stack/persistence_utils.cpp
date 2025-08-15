#include "tda/vector_stack/persistence_diagram.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <set>

namespace tda::vector_stack::utils {

PersistenceDiagram pairsToDiagram(const std::vector<tda::core::PersistencePair>& pairs) {
    return PersistenceDiagram(pairs);
}

Barcode pairsToBarcode(const std::vector<tda::core::PersistencePair>& pairs) {
    return Barcode(pairs);
}

Barcode diagramToBarcode(const PersistenceDiagram& diagram) {
    return Barcode(diagram);
}

PersistenceDiagram barcodeToDiagram(const Barcode& barcode) {
    return barcode.toPersistenceDiagram();
}

double wassersteinDistance(const PersistenceDiagram& diagram1, const PersistenceDiagram& diagram2, double p) {
    // This is a simplified implementation of the Wasserstein distance
    // In practice, you would want to use a more sophisticated algorithm
    // like the Hungarian algorithm or a specialized TDA library
    
    if (diagram1.empty() && diagram2.empty()) return 0.0;
    if (diagram1.empty()) return std::numeric_limits<double>::infinity();
    if (diagram2.empty()) return std::numeric_limits<double>::infinity();
    
    const auto& points1 = diagram1.getPoints();
    const auto& points2 = diagram2.getPoints();
    
    // For simplicity, we'll compute a basic distance metric
    // This is NOT the true Wasserstein distance, but gives a reasonable approximation
    double totalDistance = 0.0;
    size_t totalPairs = 0;
    
    // Match each point in diagram1 to the closest point in diagram2
    for (const auto& point1 : points1) {
        double minDistance = std::numeric_limits<double>::infinity();
        
        for (const auto& point2 : points2) {
            if (point1.dimension == point2.dimension) {
                // Compute distance between points
                double birthDiff = point1.birth - point2.birth;
                double deathDiff = point1.death - point2.death;
                
                // Handle infinite death times
                if (point1.isInfinite() && point2.isInfinite()) {
                    deathDiff = 0.0;
                } else if (point1.isInfinite()) {
                    deathDiff = 1.0; // Penalty for infinite vs finite
                } else if (point2.isInfinite()) {
                    deathDiff = 1.0; // Penalty for finite vs infinite
                }
                
                double distance = std::pow(std::abs(birthDiff), p) + std::pow(std::abs(deathDiff), p);
                distance = std::pow(distance, 1.0 / p);
                
                minDistance = std::min(minDistance, distance);
            }
        }
        
        if (std::isfinite(minDistance)) {
            totalDistance += minDistance;
            totalPairs++;
        }
    }
    
    // Add penalty for unmatched points
    if (points1.size() > points2.size()) {
        totalDistance += (points1.size() - points2.size()) * 1.0; // Penalty for extra points
    }
    
    return totalPairs > 0 ? totalDistance / totalPairs : 0.0;
}

double bottleneckDistance(const PersistenceDiagram& diagram1, const PersistenceDiagram& diagram2) {
    // This is a simplified implementation of the bottleneck distance
    // In practice, you would want to use a more sophisticated algorithm
    // like the Hungarian algorithm or a specialized TDA library
    
    if (diagram1.empty() && diagram2.empty()) return 0.0;
    if (diagram1.empty()) return std::numeric_limits<double>::infinity();
    if (diagram2.empty()) return std::numeric_limits<double>::infinity();
    
    const auto& points1 = diagram1.getPoints();
    const auto& points2 = diagram2.getPoints();
    
    // For simplicity, we'll compute the maximum distance between matched points
    // This is NOT the true bottleneck distance, but gives a reasonable approximation
    double maxDistance = 0.0;
    size_t matchedPairs = 0;
    
    // Match each point in diagram1 to the closest point in diagram2
    for (const auto& point1 : points1) {
        double minDistance = std::numeric_limits<double>::infinity();
        
        for (const auto& point2 : points2) {
            if (point1.dimension == point2.dimension) {
                // Compute distance between points
                double birthDiff = point1.birth - point2.birth;
                double deathDiff = point1.death - point2.death;
                
                // Handle infinite death times
                if (point1.isInfinite() && point2.isInfinite()) {
                    deathDiff = 0.0;
                } else if (point1.isInfinite()) {
                    deathDiff = 1.0; // Penalty for infinite vs finite
                } else if (point2.isInfinite()) {
                    deathDiff = 1.0; // Penalty for finite vs infinite
                }
                
                // Use L-infinity norm (max of absolute differences)
                double distance = std::max(std::abs(birthDiff), std::abs(deathDiff));
                
                minDistance = std::min(minDistance, distance);
            }
        }
        
        if (std::isfinite(minDistance)) {
            maxDistance = std::max(maxDistance, minDistance);
            matchedPairs++;
        }
    }
    
    // Add penalty for unmatched points
    if (points1.size() > points2.size()) {
        maxDistance = std::max(maxDistance, 1.0); // Penalty for extra points
    }
    
    return maxDistance;
}

/**
 * @brief Compute Betti numbers at a specific filtration value
 * 
 * This function computes the Betti numbers βₖ(epsilon) for each dimension k
 * by counting the number of persistence intervals [b, d) such that b <= epsilon < d.
 * 
 * @param diagram The persistence diagram to analyze
 * @param epsilon The filtration value at which to compute Betti numbers
 * @return Vector of Betti numbers indexed by dimension
 */
std::vector<size_t> computeBettiNumbersAtFiltration(const PersistenceDiagram& diagram, double epsilon) {
    if (diagram.empty()) return {};
    
    // Find maximum dimension
    int maxDim = 0;
    for (const auto& point : diagram.getPoints()) {
        maxDim = std::max(maxDim, point.dimension);
    }
    
    std::vector<size_t> bettiNumbers(maxDim + 1, 0);
    
    for (const auto& point : diagram.getPoints()) {
        // Check if the feature is alive at epsilon: b <= epsilon < d
        if (point.birth <= epsilon && epsilon < point.death) {
            if (point.dimension >= 0 && point.dimension <= maxDim) {
                bettiNumbers[point.dimension]++;
            }
        }
    }
    
    return bettiNumbers;
}

/**
 * @brief Compute Betti numbers at a specific filtration value for barcodes
 * 
 * @param barcode The persistence barcode to analyze
 * @param epsilon The filtration value at which to compute Betti numbers
 * @return Vector of Betti numbers indexed by dimension
 */
std::vector<size_t> computeBettiNumbersAtFiltration(const Barcode& barcode, double epsilon) {
    if (barcode.empty()) return {};
    
    // Find maximum dimension
    int maxDim = 0;
    for (const auto& interval : barcode.getIntervals()) {
        maxDim = std::max(maxDim, interval.dimension);
    }
    
    std::vector<size_t> bettiNumbers(maxDim + 1, 0);
    
    for (const auto& interval : barcode.getIntervals()) {
        // Check if the feature is alive at epsilon: b <= epsilon < d
        if (interval.birth <= epsilon && epsilon < interval.death) {
            if (interval.dimension >= 0 && interval.dimension <= maxDim) {
                bettiNumbers[interval.dimension]++;
            }
        }
    }
    
    return bettiNumbers;
}

/**
 * @brief Compute Betti numbers at multiple filtration values
 * 
 * This function computes Betti numbers across a range of filtration values,
 * useful for creating Betti number curves or understanding how topology
 * changes across the filtration.
 * 
 * @param diagram The persistence diagram to analyze
 * @param epsilonValues Vector of filtration values to evaluate
 * @return Vector of Betti number vectors, one for each epsilon value
 */
std::vector<std::vector<size_t>> computeBettiNumbersAtMultipleFiltrations(
    const PersistenceDiagram& diagram, 
    const std::vector<double>& epsilonValues) {
    
    std::vector<std::vector<size_t>> allBettiNumbers;
    allBettiNumbers.reserve(epsilonValues.size());
    
    for (double epsilon : epsilonValues) {
        allBettiNumbers.push_back(computeBettiNumbersAtFiltration(diagram, epsilon));
    }
    
    return allBettiNumbers;
}

/**
 * @brief Compute Betti numbers at multiple filtration values for barcodes
 * 
 * @param barcode The persistence barcode to analyze
 * @param epsilonValues Vector of filtration values to evaluate
 * @return Vector of Betti number vectors, one for each epsilon value
 */
std::vector<std::vector<size_t>> computeBettiNumbersAtMultipleFiltrations(
    const Barcode& barcode, 
    const std::vector<double>& epsilonValues) {
    
    std::vector<std::vector<size_t>> allBettiNumbers;
    allBettiNumbers.reserve(epsilonValues.size());
    
    for (double epsilon : epsilonValues) {
        allBettiNumbers.push_back(computeBettiNumbersAtFiltration(barcode, epsilon));
    }
    
    return allBettiNumbers;
}

/**
 * @brief Compute Betti number curve (Betti numbers vs filtration value)
 * 
 * This function creates a smooth curve of Betti numbers by evaluating
 * at many filtration values across the range of the diagram.
 * 
 * @param diagram The persistence diagram to analyze
 * @param numPoints Number of points to sample in the curve
 * @return Pair of vectors: (epsilon_values, betti_numbers_by_dimension)
 */
std::pair<std::vector<double>, std::vector<std::vector<size_t>>> computeBettiNumberCurve(
    const PersistenceDiagram& diagram, 
    size_t numPoints) {
    
    if (diagram.empty()) return {{}, {}};
    
    // Get the range of filtration values
    double minBirth = diagram.getMinBirth();
    double maxDeath = diagram.getMaxDeath();
    
    // Create evenly spaced epsilon values
    std::vector<double> epsilonValues;
    epsilonValues.reserve(numPoints);
    
    if (numPoints == 1) {
        epsilonValues.push_back(minBirth);
    } else {
        double step = (maxDeath - minBirth) / (numPoints - 1);
        for (size_t i = 0; i < numPoints; ++i) {
            epsilonValues.push_back(minBirth + i * step);
        }
    }
    
    // Compute Betti numbers at each epsilon value
    auto bettiNumbers = computeBettiNumbersAtMultipleFiltrations(diagram, epsilonValues);
    
    return {epsilonValues, bettiNumbers};
}

/**
 * @brief Compute Betti number curve for barcodes
 * 
 * @param barcode The persistence barcode to analyze
 * @param numPoints Number of points to sample in the curve
 * @return Pair of vectors: (epsilon_values, betti_numbers_by_dimension)
 */
std::pair<std::vector<double>, std::vector<std::vector<size_t>>> computeBettiNumberCurve(
    const Barcode& barcode, 
    size_t numPoints) {
    
    if (barcode.empty()) return {{}, {}};
    
    // Get the range of filtration values
    double minBirth = barcode.getMinBirth();
    double maxDeath = barcode.getMaxDeath();
    
    // Create evenly spaced epsilon values
    std::vector<double> epsilonValues;
    epsilonValues.reserve(numPoints);
    
    if (numPoints == 1) {
        epsilonValues.push_back(minBirth);
    } else {
        double step = (maxDeath - minBirth) / (numPoints - 1);
        for (size_t i = 0; i < numPoints; ++i) {
            epsilonValues.push_back(minBirth + i * step);
        }
    }
    
    // Compute Betti numbers at each epsilon value
    auto bettiNumbers = computeBettiNumbersAtMultipleFiltrations(barcode, epsilonValues);
    
    return {epsilonValues, bettiNumbers};
}

/**
 * @brief Find critical filtration values where Betti numbers change
 * 
 * This function identifies the epsilon values where the topology
 * changes (i.e., where Betti numbers change).
 * 
 * @param diagram The persistence diagram to analyze
 * @return Vector of critical epsilon values
 */
std::vector<double> findCriticalFiltrationValues(const PersistenceDiagram& diagram) {
    if (diagram.empty()) return {};
    
    std::set<double> criticalValues;
    
    // Add all birth and death times as potential critical values
    for (const auto& point : diagram.getPoints()) {
        criticalValues.insert(point.birth);
        if (std::isfinite(point.death)) {
            criticalValues.insert(point.death);
        }
    }
    
    return std::vector<double>(criticalValues.begin(), criticalValues.end());
}

/**
 * @brief Find critical filtration values for barcodes
 * 
 * @param barcode The persistence barcode to analyze
 * @return Vector of critical epsilon values
 */
std::vector<double> findCriticalFiltrationValues(const Barcode& barcode) {
    if (barcode.empty()) return {};
    
    std::set<double> criticalValues;
    
    // Add all birth and death times as potential critical values
    for (const auto& interval : barcode.getIntervals()) {
        criticalValues.insert(interval.birth);
        if (std::isfinite(interval.death)) {
            criticalValues.insert(interval.death);
        }
    }
    
    return std::vector<double>(criticalValues.begin(), criticalValues.end());
}

} // namespace tda::vector_stack::utils
