# TDA Vectorization and Storage Example

This example demonstrates the implementation of vectorization techniques for topological data analysis (TDA) and storage services for persistence diagrams and their vectorized representations.

## Overview

The example showcases:

1. Computation of persistence diagrams using Vietoris-Rips complexes
2. Vectorization of persistence diagrams using three different methods:
   - Betti Curves
   - Persistence Landscapes
   - Persistence Images
3. Storage of persistence diagrams and vectorizations using:
   - PostgreSQL (for metadata)
   - MongoDB (for raw data)
   - Hybrid storage approach combining both

## Building the Example

### Using the Direct Build Script

The simplest way to build the example is using the direct build script:

```bash
./direct_build.sh
```

This script compiles the example directly using g++ without relying on CMake.

### Running the Example

After building, run the example using:

```bash
./build/direct/vectorization_example
```

## Code Structure

The example is implemented in a single, self-contained C++ file with the following structure:

- `tda::core`: Contains the core TDA structures and algorithms
  - `PersistencePair`: Simple representation of a persistence pair
  - `PersistenceDiagram`: Collection of persistence pairs
  - `PersistentHomology`: Class for computing persistence diagrams

- `tda::vectorization`: Contains vectorization methods
  - `Vectorizer`: Base class for all vectorizers
  - `BettiCurve`: Implementation of Betti curves
  - `PersistenceLandscape`: Implementation of persistence landscapes
  - `PersistenceImage`: Implementation of persistence images
  - `VectorizerRegistry`: Factory pattern for creating vectorizers

- `tda::storage`: Contains storage services
  - `StorageService`: Base class for storage services
  - `PostgreSQLStorage`: Service for storing in PostgreSQL
  - `MongoDBStorage`: Service for storing in MongoDB
  - `HybridStorageService`: Combined storage in multiple services

## Expected Output

The example generates a random point cloud, computes its persistence diagram, vectorizes it using different methods, and simulates storing the results in databases. The output includes information about the computed persistence pairs, the dimensions of the vectorized representations, and confirmation of storage operations.

## Extending the Example

This simplified example is designed to demonstrate the concepts without external dependencies. In a real-world implementation, you would:

1. Replace the simulated persistence diagram calculation with actual TDA algorithms
2. Implement complete vectorization methods with accurate mathematics
3. Set up real database connections
4. Add error handling and performance optimizations

## Requirements

- C++17 compatible compiler (g++ or clang++)
- Standard C++ libraries
