#include "tda/vector_stack/persistence_diagram.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>


namespace tda::vector_stack {

// PersistenceDiagram implementation

PersistenceDiagram::PersistenceDiagram(const std::vector<DiagramPoint>& points) : points_(points) {
    for (const auto& point : points_) {
        validatePoint(point);
    }
}

PersistenceDiagram::PersistenceDiagram(const std::vector<tda::core::PersistencePair>& pairs) {
    for (const auto& pair : pairs) {
        DiagramPoint point;
        point.birth = pair.birth;
        point.death = pair.death;
        point.dimension = pair.dimension;
        point.birth_simplex = pair.birth_simplex;
        point.death_simplex = pair.death_simplex;
        addPoint(point);
    }
}

void PersistenceDiagram::addPoint(const DiagramPoint& point) {
    validatePoint(point);
    points_.push_back(point);
}

void PersistenceDiagram::addPoint(double birth, double death, int dimension, size_t birth_simplex, size_t death_simplex) {
    DiagramPoint point(birth, death, dimension, birth_simplex, death_simplex);
    addPoint(point);
}

void PersistenceDiagram::removePoint(size_t index) {
    if (index >= points_.size()) {
        throw std::out_of_range("Index out of range");
    }
    points_.erase(points_.begin() + index);
}

void PersistenceDiagram::clear() {
    points_.clear();
}

std::vector<PersistenceDiagram::DiagramPoint> PersistenceDiagram::getPointsByDimension(int dimension) const {
    std::vector<DiagramPoint> result;
    for (const auto& point : points_) {
        if (point.dimension == dimension) {
            result.push_back(point);
        }
    }
    return result;
}

std::vector<PersistenceDiagram::DiagramPoint> PersistenceDiagram::getFinitePoints() const {
    std::vector<DiagramPoint> result;
    for (const auto& point : points_) {
        if (point.isFinite()) {
            result.push_back(point);
        }
    }
    return result;
}

std::vector<PersistenceDiagram::DiagramPoint> PersistenceDiagram::getInfinitePoints() const {
    std::vector<DiagramPoint> result;
    for (const auto& point : points_) {
        if (point.isInfinite()) {
            result.push_back(point);
        }
    }
    return result;
}

std::vector<PersistenceDiagram::DiagramPoint> PersistenceDiagram::getPointsInRange(double minBirth, double maxBirth, double minDeath, double maxDeath) const {
    std::vector<DiagramPoint> result;
    
    // CRITICAL FIX: Use epsilon-based comparison for floating point values
    const double epsilon = std::numeric_limits<double>::epsilon() * 1000.0; // 1000x machine epsilon
    
    for (const auto& point : points_) {
        // Use epsilon-based comparison to handle floating point precision issues
        if (point.birth >= (minBirth - epsilon) && point.birth <= (maxBirth + epsilon) &&
            point.death >= (minDeath - epsilon) && point.death <= (maxDeath + epsilon)) {
            result.push_back(point);
        }
    }
    return result;
}

double PersistenceDiagram::getMinBirth() const {
    if (points_.empty()) return 0.0;
    return std::min_element(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.birth < b.birth; })->birth;
}

double PersistenceDiagram::getMaxBirth() const {
    if (points_.empty()) return 0.0;
    return std::max_element(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.birth < b.birth; })->birth;
}

double PersistenceDiagram::getMinDeath() const {
    if (points_.empty()) return 0.0;
    return std::min_element(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.death < b.death; })->death;
}

double PersistenceDiagram::getMaxDeath() const {
    if (points_.empty()) return 0.0;
    return std::max_element(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.death < b.death; })->death;
}

double PersistenceDiagram::getMinPersistence() const {
    if (points_.empty()) return 0.0;
    double minPersistence = std::numeric_limits<double>::infinity();
    for (const auto& point : points_) {
        if (point.isFinite()) {
            minPersistence = std::min(minPersistence, point.getPersistence());
        }
    }
    return std::isfinite(minPersistence) ? minPersistence : 0.0;
}

double PersistenceDiagram::getMaxPersistence() const {
    if (points_.empty()) return 0.0;
    double maxPersistence = 0.0;
    for (const auto& point : points_) {
        if (point.isFinite()) {
            maxPersistence = std::max(maxPersistence, point.getPersistence());
        }
    }
    return maxPersistence;
}

double PersistenceDiagram::getAveragePersistence() const {
    if (points_.empty()) return 0.0;
    
    double totalPersistence = 0.0;
    size_t finiteCount = 0;
    
    for (const auto& point : points_) {
        if (point.isFinite()) {
            totalPersistence += point.getPersistence();
            finiteCount++;
        }
    }
    
    return finiteCount > 0 ? totalPersistence / finiteCount : 0.0;
}

std::vector<size_t> PersistenceDiagram::getBettiNumbers() const {
    if (points_.empty()) return {};
    
    // Find maximum dimension
    int maxDim = 0;
    for (const auto& point : points_) {
        maxDim = std::max(maxDim, point.dimension);
    }
    
    std::vector<size_t> bettiNumbers(maxDim + 1, 0);
    for (const auto& point : points_) {
        if (point.dimension >= 0 && point.dimension <= maxDim) {
            bettiNumbers[point.dimension]++;
        }
    }
    
    return bettiNumbers;
}

size_t PersistenceDiagram::getBettiNumber(int dimension) const {
    if (dimension < 0) return 0;
    
    size_t count = 0;
    for (const auto& point : points_) {
        if (point.dimension == dimension) {
            count++;
        }
    }
    return count;
}

void PersistenceDiagram::sortByBirth() {
    std::sort(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.birth < b.birth; });
}

void PersistenceDiagram::sortByDeath() {
    std::sort(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.death < b.death; });
}

void PersistenceDiagram::sortByPersistence() {
    std::sort(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.getPersistence() < b.getPersistence(); });
}

void PersistenceDiagram::sortByDimension() {
    std::sort(points_.begin(), points_.end(),
        [](const DiagramPoint& a, const DiagramPoint& b) { return a.dimension < b.dimension; });
}



std::string PersistenceDiagram::toCSV() const {
    std::ostringstream oss;
    oss << "birth,death,dimension,birth_simplex,death_simplex,persistence\n";
    
    for (const auto& point : points_) {
        oss << point.birth << ","
            << point.death << ","
            << point.dimension << ","
            << point.birth_simplex << ","
            << point.death_simplex << ","
            << point.getPersistence() << "\n";
    }
    
    return oss.str();
}

std::string PersistenceDiagram::toTXT() const {
    std::ostringstream oss;
    oss << "Persistence Diagram\n";
    oss << "==================\n";
    oss << "Total points: " << points_.size() << "\n\n";
    
    oss << "Statistics:\n";
    oss << "  Birth range: [" << getMinBirth() << ", " << getMaxBirth() << "]\n";
    oss << "  Death range: [" << getMinDeath() << ", " << getMaxDeath() << "]\n";
    oss << "  Persistence range: [" << getMinPersistence() << ", " << getMaxPersistence() << "]\n";
    oss << "  Average persistence: " << getAveragePersistence() << "\n\n";
    
    oss << "Points (birth, death, dimension, birth_simplex, death_simplex, persistence):\n";
    for (const auto& point : points_) {
        oss << "  (" << point.birth << ", " << point.death << ", " << point.dimension
            << ", " << point.birth_simplex << ", " << point.death_simplex
            << ", " << point.getPersistence() << ")\n";
    }
    
    return oss.str();
}



PersistenceDiagram PersistenceDiagram::fromCSV(const std::string& csv) {
    std::istringstream iss(csv);
    std::string line;
    std::vector<DiagramPoint> points;
    
    // Skip header
    std::getline(iss, line);
    
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        
        std::istringstream lineStream(line);
        std::string token;
        DiagramPoint point;
        
        // Parse birth
        std::getline(lineStream, token, ',');
        point.birth = std::stod(token);
        
        // Parse death
        std::getline(lineStream, token, ',');
        point.death = std::stod(token);
        
        // Parse dimension
        std::getline(lineStream, token, ',');
        point.dimension = std::stoi(token);
        
        // Parse birth_simplex
        std::getline(lineStream, token, ',');
        point.birth_simplex = std::stoull(token);
        
        // Parse death_simplex
        std::getline(lineStream, token, ',');
        point.death_simplex = std::stoull(token);
        
        points.push_back(point);
    }
    
    return PersistenceDiagram(points);
}

PersistenceDiagram PersistenceDiagram::fromTXT(const std::string& txt) {
    // This is a simplified parser for the text format
    // In practice, you might want a more robust parser
    std::istringstream iss(txt);
    std::string line;
    std::vector<DiagramPoint> points;
    
    // Skip header lines
    for (int i = 0; i < 8; ++i) {
        std::getline(iss, line);
    }
    
    while (std::getline(iss, line)) {
        if (line.empty() || line.find("(") == std::string::npos) continue;
        
        // Simple parsing for format: (birth, death, dimension, birth_simplex, death_simplex, persistence)
        size_t start = line.find("(");
        size_t end = line.find(")");
        if (start == std::string::npos || end == std::string::npos) continue;
        
        std::string content = line.substr(start + 1, end - start - 1);
        std::istringstream contentStream(content);
        std::string token;
        
        DiagramPoint point;
        
        // Parse birth
        std::getline(contentStream, token, ',');
        point.birth = std::stod(token);
        
        // Parse death
        std::getline(contentStream, token, ',');
        point.death = std::stod(token);
        
        // Parse dimension
        std::getline(contentStream, token, ',');
        point.dimension = std::stoi(token);
        
        // Parse birth_simplex
        std::getline(contentStream, token, ',');
        point.birth_simplex = std::stoull(token);
        
        // Parse death_simplex
        std::getline(contentStream, token, ',');
        point.death_simplex = std::stoull(token);
        
        points.push_back(point);
    }
    
    return PersistenceDiagram(points);
}

void PersistenceDiagram::normalize(double minValue, double maxValue) {
    if (points_.empty()) return;
    
    double minBirth = getMinBirth();
    double maxBirth = getMaxBirth();
    double minDeath = getMinDeath();
    double maxDeath = getMaxDeath();
    
    double birthRange = maxBirth - minBirth;
    double deathRange = maxDeath - minDeath;
    
    if (birthRange == 0.0) birthRange = 1.0;
    if (deathRange == 0.0) deathRange = 1.0;
    
    for (auto& point : points_) {
        point.birth = minValue + (maxValue - minValue) * (point.birth - minBirth) / birthRange;
        point.death = minValue + (maxValue - minValue) * (point.death - minDeath) / deathRange;
    }
}

void PersistenceDiagram::filterByPersistence(double minPersistence) {
    points_.erase(
        std::remove_if(points_.begin(), points_.end(),
            [minPersistence](const DiagramPoint& point) {
                return point.getPersistence() < minPersistence;
            }),
        points_.end()
    );
}

void PersistenceDiagram::filterByDimension(int minDimension, int maxDimension) {
    points_.erase(
        std::remove_if(points_.begin(), points_.end(),
            [minDimension, maxDimension](const DiagramPoint& point) {
                return point.dimension < minDimension || point.dimension > maxDimension;
            }),
        points_.end()
    );
}

void PersistenceDiagram::validatePoint(const DiagramPoint& point) const {
    if (point.birth < 0.0) {
        throw std::invalid_argument("Birth time cannot be negative");
    }
    if (point.death < point.birth && std::isfinite(point.death)) {
        throw std::invalid_argument("Death time cannot be less than birth time");
    }
    if (point.dimension < 0) {
        throw std::invalid_argument("Dimension cannot be negative");
    }
}

void PersistenceDiagram::updateStatistics() {
    // Statistics are computed on-demand, so this method is currently a no-op
    // In a more complex implementation, you might want to cache statistics
}

} // namespace tda::vector_stack
