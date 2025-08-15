# TDA Vectorization and Storage Examples

This directory contains examples demonstrating the implementation of vectorization techniques for persistence diagrams in topological data analysis (TDA) and their storage in database systems.

## Available Examples

### 1. Simplified C++ Implementation

The `vectorization_storage_example_simplified.cpp` file provides a standalone C++ implementation that demonstrates:

- Representation of persistence diagrams
- Vectorization using Betti curves, persistence landscapes, and persistence images
- Storage in PostgreSQL (for metadata) and MongoDB (for raw data) using a hybrid approach
- Use of the factory pattern for creating vectorizers dynamically

This example is fully self-contained and doesn't rely on external libraries. It can be built using the `direct_build.sh` script and run using the `run_example_with_explanation.sh` script for a guided walkthrough.

### 2. Markdown Documentation Example

The `vectorization_storage_example.md` file provides a documentation-style example with detailed explanations and code snippets. This is intended as a reference rather than a runnable example.

## Building and Running

To build the simplified C++ example:

```bash
./direct_build.sh
```

To run the example with explanatory comments:

```bash
./run_example_with_explanation.sh
```

## Conceptual Overview

### Persistence Diagrams

Persistence diagrams are multisets of points in the plane, where each point (birth, death) represents a topological feature (e.g., connected component, loop, void) that appears at the "birth" value and disappears at the "death" value.

### Vectorization Methods

1. **Betti Curves**: Count the number of features of each homology dimension at different filtration values.
2. **Persistence Landscapes**: Convert the diagram into a sequence of envelope functions.
3. **Persistence Images**: Create a pixelated representation of the persistence diagram.

### Storage Strategy

The hybrid storage approach demonstrated here uses:

- **PostgreSQL**: For storing metadata about persistence diagrams and vectorizations.
- **MongoDB**: For storing the actual persistence diagrams and vector data.

This approach allows for efficient querying of metadata while maintaining the ability to store complex data structures.

## Next Steps

After understanding these examples, you might want to:

1. Integrate with actual TDA libraries like GUDHI or Dionysus
2. Implement additional vectorization methods
3. Set up real database connections
4. Develop visualization tools for the persistence diagrams and vectorizations
5. Use the vectorized representations for machine learning tasks
