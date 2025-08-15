#include "tda/vector_stack/persistence_diagram.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>


namespace tda::vector_stack {

// Barcode implementation

Barcode::Barcode(const std::vector<BarcodeInterval>& intervals) : intervals_(intervals) {
    for (const auto& interval : intervals_) {
        validateInterval(interval);
    }
}

Barcode::Barcode(const std::vector<tda::core::PersistencePair>& pairs) {
    for (const auto& pair : pairs) {
        BarcodeInterval interval;
        interval.birth = pair.birth;
        interval.death = pair.death;
        interval.dimension = pair.dimension;
        interval.birth_simplex = pair.birth_simplex;
        interval.death_simplex = pair.death_simplex;
        addInterval(interval);
    }
}

Barcode::Barcode(const PersistenceDiagram& diagram) {
    for (const auto& point : diagram.getPoints()) {
        BarcodeInterval interval;
        interval.birth = point.birth;
        interval.death = point.death;
        interval.dimension = point.dimension;
        interval.birth_simplex = point.birth_simplex;
        interval.death_simplex = point.death_simplex;
        addInterval(interval);
    }
}

void Barcode::addInterval(const BarcodeInterval& interval) {
    validateInterval(interval);
    intervals_.push_back(interval);
}

void Barcode::addInterval(double birth, double death, int dimension, size_t birth_simplex, size_t death_simplex) {
    BarcodeInterval interval(birth, death, dimension, birth_simplex, death_simplex);
    addInterval(interval);
}

void Barcode::removeInterval(size_t index) {
    if (index >= intervals_.size()) {
        throw std::out_of_range("Index out of range");
    }
    intervals_.erase(intervals_.begin() + index);
}

void Barcode::clear() {
    intervals_.clear();
}

std::vector<Barcode::BarcodeInterval> Barcode::getIntervalsByDimension(int dimension) const {
    std::vector<BarcodeInterval> result;
    for (const auto& interval : intervals_) {
        if (interval.dimension == dimension) {
            result.push_back(interval);
        }
    }
    return result;
}

std::vector<Barcode::BarcodeInterval> Barcode::getFiniteIntervals() const {
    std::vector<BarcodeInterval> result;
    for (const auto& interval : intervals_) {
        if (interval.isFinite()) {
            result.push_back(interval);
        }
    }
    return result;
}

std::vector<Barcode::BarcodeInterval> Barcode::getInfiniteIntervals() const {
    std::vector<BarcodeInterval> result;
    for (const auto& interval : intervals_) {
        if (interval.isInfinite()) {
            result.push_back(interval);
        }
    }
    return result;
}

std::vector<Barcode::BarcodeInterval> Barcode::getIntervalsInRange(double minBirth, double maxBirth, double minDeath, double maxDeath) const {
    std::vector<BarcodeInterval> result;
    for (const auto& interval : intervals_) {
        if (interval.birth >= minBirth && interval.birth <= maxBirth &&
            interval.death >= minDeath && interval.death <= maxDeath) {
            result.push_back(interval);
        }
    }
    return result;
}

double Barcode::getMinBirth() const {
    if (intervals_.empty()) return 0.0;
    return std::min_element(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.birth < b.birth; })->birth;
}

double Barcode::getMaxBirth() const {
    if (intervals_.empty()) return 0.0;
    return std::max_element(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.birth < b.birth; })->birth;
}

double Barcode::getMinDeath() const {
    if (intervals_.empty()) return 0.0;
    return std::min_element(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.death < b.death; })->death;
}

double Barcode::getMaxDeath() const {
    if (intervals_.empty()) return 0.0;
    return std::max_element(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.death < b.death; })->death;
}

double Barcode::getMinLength() const {
    if (intervals_.empty()) return 0.0;
    double minLength = std::numeric_limits<double>::infinity();
    for (const auto& interval : intervals_) {
        if (interval.isFinite()) {
            minLength = std::min(minLength, interval.getLength());
        }
    }
    return std::isfinite(minLength) ? minLength : 0.0;
}

double Barcode::getMaxLength() const {
    if (intervals_.empty()) return 0.0;
    double maxLength = 0.0;
    for (const auto& interval : intervals_) {
        if (interval.isFinite()) {
            maxLength = std::max(maxLength, interval.getLength());
        }
    }
    return maxLength;
}

double Barcode::getAverageLength() const {
    if (intervals_.empty()) return 0.0;
    
    double totalLength = 0.0;
    size_t finiteCount = 0;
    
    for (const auto& interval : intervals_) {
        if (interval.isFinite()) {
            totalLength += interval.getLength();
            finiteCount++;
        }
    }
    
    return finiteCount > 0 ? totalLength / finiteCount : 0.0;
}

std::vector<size_t> Barcode::getBettiNumbers() const {
    if (intervals_.empty()) return {};
    
    // Find maximum dimension
    int maxDim = 0;
    for (const auto& interval : intervals_) {
        maxDim = std::max(maxDim, interval.dimension);
    }
    
    std::vector<size_t> bettiNumbers(maxDim + 1, 0);
    for (const auto& interval : intervals_) {
        if (interval.dimension >= 0 && interval.dimension <= maxDim) {
            bettiNumbers[interval.dimension]++;
        }
    }
    
    return bettiNumbers;
}

size_t Barcode::getBettiNumber(int dimension) const {
    if (dimension < 0) return 0;
    
    size_t count = 0;
    for (const auto& interval : intervals_) {
        if (interval.dimension == dimension) {
            count++;
        }
    }
    return count;
}

void Barcode::sortByBirth() {
    std::sort(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.birth < b.birth; });
}

void Barcode::sortByDeath() {
    std::sort(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.death < b.death; });
}

void Barcode::sortByLength() {
    std::sort(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.getLength() < b.getLength(); });
}

void Barcode::sortByDimension() {
    std::sort(intervals_.begin(), intervals_.end(),
        [](const BarcodeInterval& a, const BarcodeInterval& b) { return a.dimension < b.dimension; });
}



std::string Barcode::toCSV() const {
    std::ostringstream oss;
    oss << "birth,death,dimension,birth_simplex,death_simplex,length\n";
    
    for (const auto& interval : intervals_) {
        oss << interval.birth << ","
            << interval.death << ","
            << interval.dimension << ","
            << interval.birth_simplex << ","
            << interval.death_simplex << ","
            << interval.getLength() << "\n";
    }
    
    return oss.str();
}

std::string Barcode::toTXT() const {
    std::ostringstream oss;
    oss << "Persistence Barcode\n";
    oss << "==================\n";
    oss << "Total intervals: " << intervals_.size() << "\n\n";
    
    oss << "Statistics:\n";
    oss << "  Birth range: [" << getMinBirth() << ", " << getMaxBirth() << "]\n";
    oss << "  Death range: [" << getMinDeath() << ", " << getMaxDeath() << "]\n";
    oss << "  Length range: [" << getMinLength() << ", " << getMaxLength() << "]\n";
    oss << "  Average length: " << getAverageLength() << "\n\n";
    
    oss << "Intervals (birth, death, dimension, birth_simplex, death_simplex, length):\n";
    for (const auto& interval : intervals_) {
        oss << "  [" << interval.birth << ", " << interval.death << "] dim=" << interval.dimension
            << " (simplices: " << interval.birth_simplex << " -> " << interval.death_simplex
            << ", length: " << interval.getLength() << ")\n";
    }
    
    return oss.str();
}



Barcode Barcode::fromCSV(const std::string& csv) {
    std::istringstream iss(csv);
    std::string line;
    std::vector<BarcodeInterval> intervals;
    
    // Skip header
    std::getline(iss, line);
    
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        
        std::istringstream lineStream(line);
        std::string token;
        BarcodeInterval interval;
        
        // Parse birth
        std::getline(lineStream, token, ',');
        interval.birth = std::stod(token);
        
        // Parse death
        std::getline(lineStream, token, ',');
        interval.death = std::stod(token);
        
        // Parse dimension
        std::getline(lineStream, token, ',');
        interval.dimension = std::stoi(token);
        
        // Parse birth_simplex
        std::getline(lineStream, token, ',');
        interval.birth_simplex = std::stoull(token);
        
        // Parse death_simplex
        std::getline(lineStream, token, ',');
        interval.death_simplex = std::stoull(token);
        
        intervals.push_back(interval);
    }
    
    return Barcode(intervals);
}

Barcode Barcode::fromTXT(const std::string& txt) {
    // This is a simplified parser for the text format
    std::istringstream iss(txt);
    std::string line;
    std::vector<BarcodeInterval> intervals;
    
    // Skip header lines
    for (int i = 0; i < 8; ++i) {
        std::getline(iss, line);
    }
    
    while (std::getline(iss, line)) {
        if (line.empty() || line.find("[") == std::string::npos) continue;
        
        // Parse format: [birth, death] dim=dimension (simplices: birth_simplex -> death_simplex, length: length)
        size_t start = line.find("[");
        size_t end = line.find("]");
        if (start == std::string::npos || end == std::string::npos) continue;
        
        std::string range = line.substr(start + 1, end - start - 1);
        std::istringstream rangeStream(range);
        std::string token;
        
        BarcodeInterval interval;
        
        // Parse birth
        std::getline(rangeStream, token, ',');
        interval.birth = std::stod(token);
        
        // Parse death
        std::getline(rangeStream, token, ',');
        interval.death = std::stod(token);
        
        // Parse dimension
        size_t dimStart = line.find("dim=");
        if (dimStart != std::string::npos) {
            size_t dimEnd = line.find(" ", dimStart);
            if (dimEnd == std::string::npos) dimEnd = line.find("(", dimStart);
            if (dimEnd != std::string::npos) {
                std::string dimStr = line.substr(dimStart + 4, dimEnd - dimStart - 4);
                interval.dimension = std::stoi(dimStr);
            }
        }
        
        // For simplicity, set simplex indices to 0
        interval.birth_simplex = 0;
        interval.death_simplex = 0;
        
        intervals.push_back(interval);
    }
    
    return Barcode(intervals);
}

void Barcode::normalize(double minValue, double maxValue) {
    if (intervals_.empty()) return;
    
    double minBirth = getMinBirth();
    double maxBirth = getMaxBirth();
    double minDeath = getMinDeath();
    double maxDeath = getMaxDeath();
    
    double birthRange = maxBirth - minBirth;
    double deathRange = maxDeath - minDeath;
    
    if (birthRange == 0.0) birthRange = 1.0;
    if (deathRange == 0.0) deathRange = 1.0;
    
    for (auto& interval : intervals_) {
        interval.birth = minValue + (maxValue - minValue) * (interval.birth - minBirth) / birthRange;
        interval.death = minValue + (maxValue - minValue) * (interval.death - minDeath) / deathRange;
    }
}

void Barcode::filterByLength(double minLength) {
    intervals_.erase(
        std::remove_if(intervals_.begin(), intervals_.end(),
            [minLength](const BarcodeInterval& interval) {
                return interval.getLength() < minLength;
            }),
        intervals_.end()
    );
}

void Barcode::filterByDimension(int minDimension, int maxDimension) {
    intervals_.erase(
        std::remove_if(intervals_.begin(), intervals_.end(),
            [minDimension, maxDimension](const BarcodeInterval& interval) {
                return interval.dimension < minDimension || interval.dimension > maxDimension;
            }),
        intervals_.end()
    );
}

PersistenceDiagram Barcode::toPersistenceDiagram() const {
    std::vector<PersistenceDiagram::DiagramPoint> points;
    
    for (const auto& interval : intervals_) {
        PersistenceDiagram::DiagramPoint point;
        point.birth = interval.birth;
        point.death = interval.death;
        point.dimension = interval.dimension;
        point.birth_simplex = interval.birth_simplex;
        point.death_simplex = interval.death_simplex;
        points.push_back(point);
    }
    
    return PersistenceDiagram(points);
}

void Barcode::validateInterval(const BarcodeInterval& interval) const {
    if (interval.birth < 0.0) {
        throw std::invalid_argument("Birth time cannot be negative");
    }
    if (interval.death < interval.birth && std::isfinite(interval.death)) {
        throw std::invalid_argument("Death time cannot be less than birth time");
    }
    if (interval.dimension < 0) {
        throw std::invalid_argument("Dimension cannot be negative");
    }
}

void Barcode::updateStatistics() {
    // Statistics are computed on-demand, so this method is currently a no-op
    // In a more complex implementation, you might want to cache statistics
}

} // namespace tda::vector_stack
