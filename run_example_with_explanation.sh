#!/bin/bash

# Run the vectorization and storage example with explanatory output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_PATH="${SCRIPT_DIR}/build/direct/vectorization_example"

# Check if the example has been built
if [ ! -f "${EXAMPLE_PATH}" ]; then
    echo "Error: Example executable not found at ${EXAMPLE_PATH}"
    echo "Please build the example first using ./direct_build.sh"
    exit 1
fi

# Display banner
echo "=================================================="
echo "  TDA Vectorization and Storage Example Runner"
echo "=================================================="
echo
echo "This script runs the TDA vectorization and storage example with"
echo "explanatory comments about what's happening at each stage."
echo
echo "Press Enter to continue..."
read -r

# Run the example
echo
echo "Starting the example execution..."
echo

# Use a temporary file to capture the output
temp_output=$(mktemp)
"${EXAMPLE_PATH}" > "${temp_output}"

# Process the output line by line with explanations
echo "------- EXECUTION WITH EXPLANATIONS -------"
echo

# Read output line by line and add explanations
while IFS= read -r line; do
    if [[ ${line} == *"TDA Platform - Vectorization and Storage Example"* ]]; then
        echo -e "\033[1;34m${line}\033[0m"  # Print header in blue
        echo "The example starts by displaying a title banner."

    elif [[ ${line} == *"Generated random point cloud"* ]]; then
        echo -e "\033[1;32m${line}\033[0m"  # Print in green
        echo "A random 3D point cloud with 100 points is generated for demonstration purposes."
        echo "In a real application, this would be your input data from sensors, measurements, etc."
        
    elif [[ ${line} == *"Computed persistent homology"* ]]; then
        echo -e "\033[1;32m${line}\033[0m"  # Print in green
        echo "The persistence diagram is computed using the Vietoris-Rips complex algorithm."
        echo "This identifies topological features (connected components, loops, voids) at different scales."
        
    elif [[ ${line} == *"Dimension: "* ]]; then
        echo -e "\033[1;33m${line}\033[0m"  # Print in yellow
        
    elif [[ ${line} == *"Vectorizing the persistence diagram"* ]]; then
        echo -e "\033[1;36m${line}\033[0m"  # Print in cyan
        echo "Now the persistence diagram will be converted to vector representations using three methods."
        
    elif [[ ${line} == *"BettiCurve: Vector dimension"* ]]; then
        echo -e "\033[1;33m${line}\033[0m"  # Print in yellow
        echo "Betti curves count the number of topological features at each filtration value."
        
    elif [[ ${line} == *"PersistenceLandscape: Vector dimension"* ]]; then
        echo -e "\033[1;33m${line}\033[0m"  # Print in yellow
        echo "Persistence landscapes convert the diagram into a sequence of envelope functions."
        
    elif [[ ${line} == *"PersistenceImage: Vector dimension"* ]]; then
        echo -e "\033[1;33m${line}\033[0m"  # Print in yellow
        echo "Persistence images create a pixelated representation of the persistence diagram."
        
    elif [[ ${line} == *"Storing results in databases"* ]]; then
        echo -e "\033[1;36m${line}\033[0m"  # Print in cyan
        echo "The persistence diagram and vectorizations are now stored in databases (simulated)."
        echo "A hybrid approach uses PostgreSQL for metadata and MongoDB for the actual data."
        
    elif [[ ${line} == *"Connecting to"* ]]; then
        echo -e "\033[1;33m${line}\033[0m"  # Print in yellow
        
    elif [[ ${line} == *"Initializing hybrid storage"* ]]; then
        echo -e "\033[1;33m${line}\033[0m"  # Print in yellow
        echo "The hybrid storage service uses both PostgreSQL and MongoDB for different aspects of storage."
        
    elif [[ ${line} == *"Using the vectorizer registry"* ]]; then
        echo -e "\033[1;36m${line}\033[0m"  # Print in cyan
        echo "The vectorizer registry demonstrates the factory pattern for creating vectorizers."
        echo "This allows dynamic selection of vectorization methods and configuration."
        
    elif [[ ${line} == *"Example completed successfully"* ]]; then
        echo -e "\033[1;32m${line}\033[0m"  # Print in green
        echo "The example has completed successfully, demonstrating the entire workflow."
        
    else
        # Print regular output
        echo "${line}"
    fi
done < "${temp_output}"

# Clean up temporary file
rm "${temp_output}"

echo
echo "=================================================="
echo "                Example Completed"
echo "=================================================="
echo
echo "This demonstration shows how persistence diagrams from topological data"
echo "analysis can be vectorized and stored in a database system."
echo
echo "Key takeaways:"
echo "  1. Persistence diagrams capture multi-scale topological features"
echo "  2. Vectorization methods convert diagrams to ML-compatible formats"
echo "  3. Different vectorization techniques have different properties"
echo "  4. A hybrid storage approach separates metadata from raw data"
echo "  5. The factory pattern allows flexible creation of vectorizers"
echo
echo "For more information, see the README.md in the examples directory."
