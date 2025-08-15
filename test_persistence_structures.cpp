#include "tda/vector_stack/persistence_diagram.hpp"
#include "tda/core/types.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>

using namespace tda::vector_stack;

void testPersistenceDiagramBasic() {
    std::cout << "Testing PersistenceDiagram basic functionality..." << std::endl;
    
    // Test default constructor
    PersistenceDiagram diagram;
    assert(diagram.empty());
    assert(diagram.size() == 0);
    
    // Test adding points
    diagram.addPoint(0.0, 1.0, 0, 1, 2);
    diagram.addPoint(0.5, 1.5, 1, 3, 4);
    diagram.addPoint(1.0, 2.0, 0, 5, 6);
    
    assert(diagram.size() == 3);
    assert(!diagram.empty());
    
    // Test accessors
    const auto& points = diagram.getPoints();
    assert(points.size() == 3);
    assert(points[0].birth == 0.0);
    assert(points[0].death == 1.0);
    assert(points[0].dimension == 0);
    assert(points[0].birth_simplex == 1);
    assert(points[0].death_simplex == 2);
    
    std::cout << "✓ Basic functionality successful" << std::endl;
}

void testPersistenceDiagramStatistics() {
    std::cout << "Testing PersistenceDiagram statistics..." << std::endl;
    
    PersistenceDiagram diagram;
    diagram.addPoint(0.0, 1.0, 0);
    diagram.addPoint(0.5, 1.5, 1);
    diagram.addPoint(1.0, 2.0, 0);
    diagram.addPoint(0.2, 1.8, 1);
    
    // Test statistics
    assert(diagram.getMinBirth() == 0.0);
    assert(diagram.getMaxBirth() == 1.0);
    assert(diagram.getMinDeath() == 1.0);
    assert(diagram.getMaxDeath() == 2.0);
    assert(diagram.getMinPersistence() == 1.0);
    assert(diagram.getMaxPersistence() == 1.6);
    
    double avgPersistence = diagram.getAveragePersistence();
    assert(avgPersistence > 1.1 && avgPersistence < 1.2); // Should be around 1.15
    
    // Test Betti numbers
    auto bettiNumbers = diagram.getBettiNumbers();
    assert(bettiNumbers.size() == 2); // Dimensions 0 and 1
    assert(bettiNumbers[0] == 2); // 2 features in dimension 0
    assert(bettiNumbers[1] == 2); // 2 features in dimension 1
    
    assert(diagram.getBettiNumber(0) == 2);
    assert(diagram.getBettiNumber(1) == 2);
    assert(diagram.getBettiNumber(2) == 0);
    
    std::cout << "✓ Statistics computation successful" << std::endl;
}

void testPersistenceDiagramFiltering() {
    std::cout << "Testing PersistenceDiagram filtering..." << std::endl;
    
    PersistenceDiagram diagram;
    diagram.addPoint(0.0, 1.0, 0);
    diagram.addPoint(0.5, 1.5, 1);
    diagram.addPoint(1.0, 2.0, 0);
    diagram.addPoint(0.2, 1.8, 1);
    
    // Test filtering by dimension
    auto dim0Points = diagram.getPointsByDimension(0);
    assert(dim0Points.size() == 2);
    for (const auto& point : dim0Points) {
        assert(point.dimension == 0);
    }
    
    auto dim1Points = diagram.getPointsByDimension(1);
    assert(dim1Points.size() == 2);
    for (const auto& point : dim1Points) {
        assert(point.dimension == 1);
    }
    
    // Test filtering by persistence
    diagram.filterByPersistence(0.6);
    assert(diagram.size() == 4); // All points have persistence >= 0.6
    
    // Test filtering by dimension range
    diagram.clear();
    diagram.addPoint(0.0, 1.0, 0);
    diagram.addPoint(0.5, 1.5, 1);
    diagram.addPoint(1.0, 2.0, 2);
    diagram.addPoint(0.2, 1.8, 1);
    
    diagram.filterByDimension(0, 1);
    assert(diagram.size() == 3); // Only dimensions 0 and 1
    
    std::cout << "✓ Filtering functionality successful" << std::endl;
}

void testPersistenceDiagramSorting() {
    std::cout << "Testing PersistenceDiagram sorting..." << std::endl;
    
    PersistenceDiagram diagram;
    diagram.addPoint(1.0, 2.0, 0);
    diagram.addPoint(0.0, 1.0, 1);
    diagram.addPoint(0.5, 1.5, 0);
    
    // Test sorting by birth
    diagram.sortByBirth();
    const auto& points = diagram.getPoints();
    assert(points[0].birth == 0.0);
    assert(points[1].birth == 0.5);
    assert(points[2].birth == 1.0);
    
    // Test sorting by death
    diagram.sortByDeath();
    assert(points[0].death == 1.0);
    assert(points[1].death == 1.5);
    assert(points[2].death == 2.0);
    
    // Test sorting by persistence
    diagram.sortByPersistence();
    assert(points[0].getPersistence() == 1.0);
    assert(points[1].getPersistence() == 1.0);
    assert(points[2].getPersistence() == 1.0);
    
    // Test sorting by dimension
    diagram.sortByDimension();
    assert(points[0].dimension == 0);
    assert(points[1].dimension == 0);
    assert(points[2].dimension == 1);
    
    std::cout << "✓ Sorting functionality successful" << std::endl;
}

void testPersistenceDiagramSerialization() {
    std::cout << "Testing PersistenceDiagram serialization..." << std::endl;
    
    PersistenceDiagram diagram;
    diagram.addPoint(0.0, 1.0, 0, 1, 2);
    diagram.addPoint(0.5, 1.5, 1, 3, 4);
    
    // Test CSV serialization
    std::string csv = diagram.toCSV();
    assert(csv.find("birth,death,dimension") != std::string::npos);
    assert(csv.find("0,1,0,1,2,1") != std::string::npos); // birth,death,dim,birth_sim,death_sim,persistence
    
    // Test TXT serialization
    std::string txt = diagram.toTXT();
    assert(txt.find("Persistence Diagram") != std::string::npos);
    assert(txt.find("Total points: 2") != std::string::npos);
    
    // Test CSV deserialization
    PersistenceDiagram diagram2 = PersistenceDiagram::fromCSV(csv);
    assert(diagram2.size() == 2);
    
    std::cout << "✓ Serialization functionality successful" << std::endl;
}

void testBarcodeBasic() {
    std::cout << "Testing Barcode basic functionality..." << std::endl;
    
    // Test default constructor
    Barcode barcode;
    assert(barcode.empty());
    assert(barcode.size() == 0);
    
    // Test adding intervals
    barcode.addInterval(0.0, 1.0, 0, 1, 2);
    barcode.addInterval(0.5, 1.5, 1, 3, 4);
    barcode.addInterval(1.0, 2.0, 0, 5, 6);
    
    assert(barcode.size() == 3);
    assert(!barcode.empty());
    
    // Test accessors
    const auto& intervals = barcode.getIntervals();
    assert(intervals.size() == 3);
    assert(intervals[0].birth == 0.0);
    assert(intervals[0].death == 1.0);
    assert(intervals[0].dimension == 0);
    assert(intervals[0].birth_simplex == 1);
    assert(intervals[0].death_simplex == 2);
    
    std::cout << "✓ Basic functionality successful" << std::endl;
}

void testBarcodeStatistics() {
    std::cout << "Testing Barcode statistics..." << std::endl;
    
    Barcode barcode;
    barcode.addInterval(0.0, 1.0, 0);
    barcode.addInterval(0.5, 1.5, 1);
    barcode.addInterval(1.0, 2.0, 0);
    barcode.addInterval(0.2, 1.8, 1);
    
    // Test statistics
    assert(barcode.getMinBirth() == 0.0);
    assert(barcode.getMaxBirth() == 1.0);
    assert(barcode.getMinDeath() == 1.0);
    assert(barcode.getMaxDeath() == 2.0);
    assert(barcode.getMinLength() == 1.0);
    assert(barcode.getMaxLength() == 1.6);
    
    double avgLength = barcode.getAverageLength();
    assert(avgLength > 1.1 && avgLength < 1.2); // Should be around 1.15
    
    // Test Betti numbers
    auto bettiNumbers = barcode.getBettiNumbers();
    assert(bettiNumbers.size() == 2); // Dimensions 0 and 1
    assert(bettiNumbers[0] == 2); // 2 features in dimension 0
    assert(bettiNumbers[1] == 2); // 2 features in dimension 1
    
    assert(barcode.getBettiNumber(0) == 2);
    assert(barcode.getBettiNumber(1) == 2);
    assert(barcode.getBettiNumber(2) == 0);
    
    std::cout << "✓ Statistics computation successful" << std::endl;
}

void testBarcodeFiltering() {
    std::cout << "Testing Barcode filtering..." << std::endl;
    
    Barcode barcode;
    barcode.addInterval(0.0, 1.0, 0);
    barcode.addInterval(0.5, 1.5, 1);
    barcode.addInterval(1.0, 2.0, 0);
    barcode.addInterval(0.2, 1.8, 1);
    
    // Test filtering by dimension
    auto dim0Intervals = barcode.getIntervalsByDimension(0);
    assert(dim0Intervals.size() == 2);
    for (const auto& interval : dim0Intervals) {
        assert(interval.dimension == 0);
    }
    
    auto dim1Intervals = barcode.getIntervalsByDimension(1);
    assert(dim1Intervals.size() == 2);
    for (const auto& interval : dim1Intervals) {
        assert(interval.dimension == 1);
    }
    
    // Test filtering by length
    barcode.filterByLength(0.6);
    assert(barcode.size() == 4); // All intervals have length >= 0.6
    
    // Test filtering by dimension range
    barcode.clear();
    barcode.addInterval(0.0, 1.0, 0);
    barcode.addInterval(0.5, 1.5, 1);
    barcode.addInterval(1.0, 2.0, 2);
    barcode.addInterval(0.2, 1.8, 1);
    
    barcode.filterByDimension(0, 1);
    assert(barcode.size() == 3); // Only dimensions 0 and 1
    
    std::cout << "✓ Filtering functionality successful" << std::endl;
}

void testBarcodeSorting() {
    std::cout << "Testing Barcode sorting..." << std::endl;
    
    Barcode barcode;
    barcode.addInterval(1.0, 2.0, 0);
    barcode.addInterval(0.0, 1.0, 1);
    barcode.addInterval(0.5, 1.5, 0);
    
    // Test sorting by birth
    barcode.sortByBirth();
    const auto& intervals = barcode.getIntervals();
    assert(intervals[0].birth == 0.0);
    assert(intervals[1].birth == 0.5);
    assert(intervals[2].birth == 1.0);
    
    // Test sorting by death
    barcode.sortByDeath();
    assert(intervals[0].death == 1.0);
    assert(intervals[1].death == 1.5);
    assert(intervals[2].death == 2.0);
    
    // Test sorting by length
    barcode.sortByLength();
    assert(intervals[0].getLength() == 1.0);
    assert(intervals[1].getLength() == 1.0);
    assert(intervals[2].getLength() == 1.0);
    
    // Test sorting by dimension
    barcode.sortByDimension();
    assert(intervals[0].dimension == 0);
    assert(intervals[1].dimension == 0);
    assert(intervals[2].dimension == 1);
    
    std::cout << "✓ Sorting functionality successful" << std::endl;
}

void testBarcodeSerialization() {
    std::cout << "Testing Barcode serialization..." << std::endl;
    
    Barcode barcode;
    barcode.addInterval(0.0, 1.0, 0, 1, 2);
    barcode.addInterval(0.5, 1.5, 1, 3, 4);
    
    // Test CSV serialization
    std::string csv = barcode.toCSV();
    assert(csv.find("birth,death,dimension") != std::string::npos);
    assert(csv.find("0,1,0,1,2,1") != std::string::npos); // birth,death,dim,birth_sim,death_sim,length
    
    // Test TXT serialization
    std::string txt = barcode.toTXT();
    assert(txt.find("Persistence Barcode") != std::string::npos);
    assert(txt.find("Total intervals: 2") != std::string::npos);
    
    // Test CSV deserialization
    Barcode barcode2 = Barcode::fromCSV(csv);
    assert(barcode2.size() == 2);
    
    std::cout << "✓ Serialization functionality successful" << std::endl;
}

void testConversionFunctions() {
    std::cout << "Testing conversion functions..." << std::endl;
    
    // Create persistence pairs
    std::vector<tda::core::PersistencePair> pairs;
    pairs.emplace_back(0, 0.0, 1.0);
    pairs.emplace_back(1, 0.5, 1.5);
    pairs.emplace_back(0, 1.0, 2.0);
    
    // Test conversion to diagram
    PersistenceDiagram diagram = utils::pairsToDiagram(pairs);
    assert(diagram.size() == 3);
    
    // Test conversion to barcode
    Barcode barcode = utils::pairsToBarcode(pairs);
    assert(barcode.size() == 3);
    
    // Test conversion between diagram and barcode
    Barcode barcode2 = utils::diagramToBarcode(diagram);
    assert(barcode2.size() == 3);
    
    PersistenceDiagram diagram2 = utils::barcodeToDiagram(barcode);
    assert(diagram2.size() == 3);
    
    // Test conversion using class methods
    Barcode barcode3(diagram);
    assert(barcode3.size() == 3);
    
    PersistenceDiagram diagram3 = barcode.toPersistenceDiagram();
    assert(diagram3.size() == 3);
    
    std::cout << "✓ Conversion functions successful" << std::endl;
}

void testDistanceFunctions() {
    std::cout << "Testing distance functions..." << std::endl;
    
    // Create two diagrams
    PersistenceDiagram diagram1;
    diagram1.addPoint(0.0, 1.0, 0);
    diagram1.addPoint(0.5, 1.5, 1);
    
    PersistenceDiagram diagram2;
    diagram2.addPoint(0.1, 1.1, 0);
    diagram2.addPoint(0.6, 1.6, 1);
    
    // Test Wasserstein distance
    double wasserstein = utils::wassersteinDistance(diagram1, diagram2, 2.0);
    assert(wasserstein >= 0.0);
    assert(wasserstein < 1.0); // Should be small for similar diagrams
    
    // Test bottleneck distance
    double bottleneck = utils::bottleneckDistance(diagram1, diagram2);
    assert(bottleneck >= 0.0);
    assert(bottleneck < 1.0); // Should be small for similar diagrams
    
    // Test distance to self (should be 0)
    double selfWasserstein = utils::wassersteinDistance(diagram1, diagram1, 2.0);
    assert(selfWasserstein == 0.0);
    
    double selfBottleneck = utils::bottleneckDistance(diagram1, diagram1);
    assert(selfBottleneck == 0.0);
    
    std::cout << "✓ Distance functions successful" << std::endl;
}

void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test empty diagrams
    PersistenceDiagram emptyDiagram;
    assert(emptyDiagram.getMinBirth() == 0.0);
    assert(emptyDiagram.getMaxBirth() == 0.0);
    assert(emptyDiagram.getMinDeath() == 0.0);
    assert(emptyDiagram.getMaxDeath() == 0.0);
    assert(emptyDiagram.getMinPersistence() == 0.0);
    assert(emptyDiagram.getMaxPersistence() == 0.0);
    assert(emptyDiagram.getAveragePersistence() == 0.0);
    
    // Test empty barcodes
    Barcode emptyBarcode;
    assert(emptyBarcode.getMinBirth() == 0.0);
    assert(emptyBarcode.getMaxBirth() == 0.0);
    assert(emptyBarcode.getMinDeath() == 0.0);
    assert(emptyBarcode.getMaxDeath() == 0.0);
    assert(emptyBarcode.getMinLength() == 0.0);
    assert(emptyBarcode.getMaxLength() == 0.0);
    assert(emptyBarcode.getAverageLength() == 0.0);
    
    // Test single point
    PersistenceDiagram singleDiagram;
    singleDiagram.addPoint(1.0, 2.0, 0);
    assert(singleDiagram.getMinBirth() == 1.0);
    assert(singleDiagram.getMaxBirth() == 1.0);
    assert(singleDiagram.getMinDeath() == 2.0);
    assert(singleDiagram.getMaxDeath() == 2.0);
    assert(singleDiagram.getMinPersistence() == 1.0);
    assert(singleDiagram.getMaxPersistence() == 1.0);
    assert(singleDiagram.getAveragePersistence() == 1.0);
    
    std::cout << "✓ Edge cases successful" << std::endl;
}

void testBettiNumberCalculation() {
    std::cout << "Testing Betti number calculation utilities..." << std::endl;
    
    // Create a simple persistence diagram for testing
    PersistenceDiagram diagram;
    diagram.addPoint(0.0, 1.0, 0);   // β₀: alive from 0.0 to 1.0
    diagram.addPoint(0.5, 1.5, 1);   // β₁: alive from 0.5 to 1.5
    diagram.addPoint(1.0, 2.0, 0);   // β₀: alive from 1.0 to 2.0
    diagram.addPoint(0.2, 1.8, 1);   // β₁: alive from 0.2 to 1.8
    diagram.addPoint(0.8, 3.0, 2);   // β₂: alive from 0.8 to 3.0
    

    
    // Test Betti numbers at specific filtration values
    auto bettiAt0 = utils::computeBettiNumbersAtFiltration(diagram, 0.0);
    assert(bettiAt0.size() == 3); // Dimensions 0, 1, 2
    assert(bettiAt0[0] == 1);     // β₀(0.0) = 1 (only first point)
    assert(bettiAt0[1] == 0);     // β₁(0.0) = 0 (no 1D features yet)
    assert(bettiAt0[2] == 0);     // β₂(0.0) = 0 (no 2D features yet)
    
    auto bettiAt05 = utils::computeBettiNumbersAtFiltration(diagram, 0.5);
    assert(bettiAt05.size() == 3);
    assert(bettiAt05[0] == 1);    // β₀(0.5) = 1 (only first point is alive)
    assert(bettiAt05[1] == 2);    // β₁(0.5) = 2 (second and fourth points are alive)
    assert(bettiAt05[2] == 0);    // β₂(0.5) = 0 (fifth point not born yet)
    
    auto bettiAt1 = utils::computeBettiNumbersAtFiltration(diagram, 1.0);
    assert(bettiAt1.size() == 3);
    assert(bettiAt1[0] == 1);     // β₀(1.0) = 1 (only third point)
    assert(bettiAt1[1] == 2);     // β₁(1.0) = 2 (second and fourth points)
    assert(bettiAt1[2] == 1);     // β₂(1.0) = 1 (fifth point)
    
    auto bettiAt2 = utils::computeBettiNumbersAtFiltration(diagram, 2.0);
    assert(bettiAt2.size() == 3);
    assert(bettiAt2[0] == 0);     // β₀(2.0) = 0 (no 0D features alive)
    assert(bettiAt2[1] == 0);     // β₁(2.0) = 0 (no 1D features alive)
    assert(bettiAt2[2] == 1);     // β₂(2.0) = 1 (only fifth point)
    
    // Test with barcode
    Barcode barcode(diagram);
    auto bettiBarcodeAt1 = utils::computeBettiNumbersAtFiltration(barcode, 1.0);
    assert(bettiBarcodeAt1.size() == 3);
    assert(bettiBarcodeAt1[0] == 1);
    assert(bettiBarcodeAt1[1] == 2);
    assert(bettiBarcodeAt1[2] == 1);
    
    // Test multiple filtration values
    std::vector<double> epsilonValues = {0.0, 0.5, 1.0, 1.5, 2.0};
    auto bettiMultiple = utils::computeBettiNumbersAtMultipleFiltrations(diagram, epsilonValues);
    assert(bettiMultiple.size() == 5);
    assert(bettiMultiple[0][0] == 1); // β₀(0.0) = 1
    assert(bettiMultiple[2][1] == 2); // β₁(1.0) = 2
    assert(bettiMultiple[4][2] == 1); // β₂(2.0) = 1
    
    // Test Betti number curve
    auto curve = utils::computeBettiNumberCurve(diagram, 5);
    assert(curve.first.size() == 5);  // 5 epsilon values
    assert(curve.second.size() == 5); // 5 Betti number vectors
    
    // Test critical filtration values
    auto criticalValues = utils::findCriticalFiltrationValues(diagram);
    assert(criticalValues.size() == 9); // 5 birth + 5 death times, but 1.0 appears twice
    assert(std::find(criticalValues.begin(), criticalValues.end(), 0.0) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 0.2) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 0.5) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 0.8) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 1.0) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 1.5) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 1.8) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 2.0) != criticalValues.end());
    assert(std::find(criticalValues.begin(), criticalValues.end(), 3.0) != criticalValues.end());
    
    // Test edge cases
    PersistenceDiagram emptyDiagram;
    auto bettiEmpty = utils::computeBettiNumbersAtFiltration(emptyDiagram, 1.0);
    assert(bettiEmpty.empty());
    
    auto criticalEmpty = utils::findCriticalFiltrationValues(emptyDiagram);
    assert(criticalEmpty.empty());
    
    // Test single point diagram
    PersistenceDiagram singleDiagram;
    singleDiagram.addPoint(1.0, 2.0, 0);
    auto bettiSingle = utils::computeBettiNumbersAtFiltration(singleDiagram, 1.5);
    assert(bettiSingle.size() == 1);
    assert(bettiSingle[0] == 1); // β₀(1.5) = 1
    
    std::cout << "✓ Betti number calculation utilities successful" << std::endl;
}

int main() {
    std::cout << "Starting PersistenceDiagram and Barcode tests..." << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        testPersistenceDiagramBasic();
        testPersistenceDiagramStatistics();
        testPersistenceDiagramFiltering();
        testPersistenceDiagramSorting();
        testPersistenceDiagramSerialization();
        
        testBarcodeBasic();
        testBarcodeStatistics();
        testBarcodeFiltering();
        testBarcodeSorting();
        testBarcodeSerialization();
        
        testConversionFunctions();
        testDistanceFunctions();
        testEdgeCases();
        testBettiNumberCalculation();
        
        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        std::cout << "PersistenceDiagram and Barcode classes are working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}
