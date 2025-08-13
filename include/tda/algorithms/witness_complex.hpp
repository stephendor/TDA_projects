#pragma once

#include "tda/core/types.hpp"
#include <vector>
#include <memory>
#include <limits>

namespace tda::algorithms {

/**
 * @brief Witness Complex approximation for large-scale topological data analysis
 * 
 * This class implements witness complex construction for approximating topological
 * features of point clouds using a small set of landmark points. This approach
 * significantly reduces computational complexity while preserving topological
 * information.
 */
class WitnessComplex {
public:
    /**
     * @brief Configuration parameters for witness complex computation
     */
    struct WitnessConfig {
        size_t num_landmarks = 50;         ///< Number of landmark points to select (reduced for performance)
        double relaxation = 0.1;           ///< Relaxation parameter for weak witness complexes
        int max_dimension = 2;             ///< Maximum simplex dimension to compute
        bool use_strong_witness = false;   ///< Use strong vs weak witness definition
        std::string landmark_strategy = "farthest_point"; ///< "farthest_point", "random", "density"
        double distance_threshold = std::numeric_limits<double>::infinity(); ///< Max distance for witness relations
        size_t min_witnesses = 1;          ///< Minimum number of witnesses required per simplex
        bool preserve_ordering = true;     ///< Preserve distance ordering in witness relations
        size_t max_witnesses_per_point = 10; ///< Limit witnesses per point for performance
    };

    /**
     * @brief Result structure for witness complex computation
     */
    struct Result {
        std::vector<std::vector<size_t>> simplices;        ///< Generated simplices (landmark indices)
        std::vector<double> filtration_values;             ///< Corresponding filtration values
        std::vector<size_t> landmark_indices;              ///< Selected landmark points
        std::vector<std::vector<size_t>> witness_relations; ///< Witness relationships for each simplex
        size_t total_witness_checks;                       ///< Total witness relationship checks performed
        double approximation_quality;                      ///< Estimated quality metric [0,1]
        double computation_time_seconds;                   ///< Total computation time
        std::string strategy_used;                         ///< Landmark strategy that was used
    };

    /**
     * @brief Construct witness complex with default configuration
     */
    WitnessComplex() = default;

    /**
     * @brief Construct witness complex with custom configuration
     * @param config Configuration parameters
     */
    explicit WitnessComplex(const WitnessConfig& config) : config_(config) {}

    /**
     * @brief Compute witness complex from point cloud
     * @param points Input point cloud as nested vectors
     * @param config Configuration parameters
     * @return Result containing witness complex or error message
     */
    tda::core::Result<Result> computeWitnessComplex(
        const std::vector<std::vector<double>>& points,
        const WitnessConfig& config
    );

    /**
     * @brief Compute witness complex with default configuration
     * @param points Input point cloud as nested vectors
     * @return Result containing witness complex or error message
     */
    tda::core::Result<Result> computeWitnessComplex(
        const std::vector<std::vector<double>>& points
    );

    /**
     * @brief Get the current configuration
     * @return Current configuration parameters
     */
    const WitnessConfig& getConfig() const { return config_; }

    /**
     * @brief Update configuration parameters
     * @param config New configuration parameters
     */
    void setConfig(const WitnessConfig& config) { config_ = config; }

private:
    WitnessConfig config_;  ///< Current configuration

    /**
     * @brief Select landmark points using various strategies
     * @param points Input point cloud
     * @param num_landmarks Number of landmarks to select
     * @param strategy Strategy to use for selection
     * @return Indices of selected landmark points
     */
    std::vector<size_t> selectLandmarks(
        const std::vector<std::vector<double>>& points,
        size_t num_landmarks,
        const std::string& strategy
    );

    /**
     * @brief Farthest point sampling for landmark selection
     * @param points Input point cloud
     * @param num_landmarks Number of landmarks to select
     * @return Indices of selected landmarks
     */
    std::vector<size_t> farthestPointSampling(
        const std::vector<std::vector<double>>& points,
        size_t num_landmarks
    );

    /**
     * @brief Random sampling for landmark selection
     * @param points Input point cloud
     * @param num_landmarks Number of landmarks to select
     * @return Indices of selected landmarks
     */
    std::vector<size_t> randomSampling(
        const std::vector<std::vector<double>>& points,
        size_t num_landmarks
    );

    /**
     * @brief Density-based sampling for landmark selection
     * @param points Input point cloud
     * @param num_landmarks Number of landmarks to select
     * @return Indices of selected landmarks
     */
    std::vector<size_t> densityBasedSampling(
        const std::vector<std::vector<double>>& points,
        size_t num_landmarks
    );

    /**
     * @brief Build witness relations between points and landmarks
     * @param points Full point cloud
     * @param landmarks Selected landmark indices
     * @param config Configuration parameters
     * @return Witness relations structure
     */
    std::vector<std::vector<std::pair<size_t, double>>> buildWitnessRelations(
        const std::vector<std::vector<double>>& points,
        const std::vector<size_t>& landmarks,
        const WitnessConfig& config
    );

    /**
     * @brief Check if a simplex has sufficient witnesses
     * @param simplex Landmark indices forming the simplex
     * @param witness_relations Precomputed witness relations
     * @param use_strong_witness Whether to use strong witness definition
     * @param min_witnesses Minimum number of witnesses required
     * @return True if simplex has sufficient witnesses
     */
    bool hasSufficientWitnesses(
        const std::vector<size_t>& simplex,
        const std::vector<std::vector<std::pair<size_t, double>>>& witness_relations,
        bool use_strong_witness,
        size_t min_witnesses
    );

    /**
     * @brief Generate simplices from witness relations
     * @param landmarks Selected landmark indices
     * @param witness_relations Witness relations structure
     * @param max_dimension Maximum simplex dimension
     * @param config Configuration parameters
     * @return Generated simplices and their filtration values
     */
    std::pair<std::vector<std::vector<size_t>>, std::vector<double>>
    generateSimplicesFromWitnesses(
        const std::vector<size_t>& landmarks,
        const std::vector<std::vector<std::pair<size_t, double>>>& witness_relations,
        int max_dimension,
        const WitnessConfig& config
    );

    /**
     * @brief Compute filtration value for a simplex
     * @param simplex Landmark indices forming the simplex
     * @param witness_relations Witness relations structure
     * @param relaxation Relaxation parameter
     * @return Filtration value for the simplex
     */
    double computeFiltrationValue(
        const std::vector<size_t>& simplex,
        const std::vector<std::vector<std::pair<size_t, double>>>& witness_relations,
        double relaxation
    );

    /**
     * @brief Compute Euclidean distance between two points
     * @param p1 First point
     * @param p2 Second point
     * @return Euclidean distance
     */
    double euclideanDistance(
        const std::vector<double>& p1,
        const std::vector<double>& p2
    ) const;

    /**
     * @brief Estimate approximation quality
     * @param num_points Total number of points
     * @param num_landmarks Number of landmarks used
     * @param num_simplices Number of simplices generated
     * @param strategy Strategy used
     * @return Quality estimate [0,1] where 1 is perfect
     */
    double estimateApproximationQuality(
        size_t num_points,
        size_t num_landmarks,
        size_t num_simplices,
        const std::string& strategy
    ) const;

    /**
     * @brief Generate all possible simplices up to given dimension
     * @param landmarks List of landmark indices
     * @param max_dimension Maximum dimension to generate
     * @return All possible simplices
     */
    std::vector<std::vector<size_t>> generateAllPossibleSimplices(
        const std::vector<size_t>& landmarks,
        int max_dimension
    );

    /**
     * @brief Generate combinations of k elements from n elements
     * @param elements Input elements
     * @param k Size of combinations
     * @return All k-combinations
     */
    std::vector<std::vector<size_t>> generateCombinations(
        const std::vector<size_t>& elements,
        size_t k
    );
};

} // namespace tda::algorithms
