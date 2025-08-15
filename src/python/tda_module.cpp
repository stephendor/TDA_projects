#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for binding functions
void bind_core(py::module_& m);
void bind_algorithms(py::module_& m);
void bind_vector_stack(py::module_& m);

PYBIND11_MODULE(tda_python, m) {
    m.doc() = R"pbdoc(
        TDA Vector Stack Python Bindings
        
        This module provides Python bindings for the TDA (Topological Data Analysis) 
        Vector Stack C++23 library. It includes:
        
        - Core TDA types and data structures
        - Point cloud representation and manipulation
        - Persistent homology algorithms (Vietoris-Rips, Alpha, Čech, DTM)
        - Vector stack operations for efficient TDA computations
        - NumPy integration for seamless data transfer
        - Result types with error handling
        
        Main Submodules:
        ---------------
        - tda_python.types: Core types, error handling, and utilities
        - tda_python.algorithms: TDA algorithms and parameter classes
        - tda_python.vector_stack: High-performance vector stack operations
        
        Example Usage:
        -------------
        >>> import numpy as np
        >>> import tda_python as tda
        >>> 
        >>> # Create point cloud from NumPy array
        >>> points = np.random.random((100, 3))
        >>> pc = tda.PointCloud.from_numpy(points)
        >>> 
        >>> # Set up Vietoris-Rips computation
        >>> params = tda.algorithms.default_vr_params()
        >>> params.max_edge_length = 1.0
        >>> params.max_dimension = 2
        >>> 
        >>> # Compute persistent homology
        >>> result = tda.algorithms.compute_vietoris_rips(pc, params)
        >>> print(f"Found {len(result.persistence_pairs)} persistence pairs")
        >>> 
        >>> # Access results
        >>> if result.is_valid():
        >>>     dim_1_pairs = result.get_dimension_pairs(1)
        >>>     print(f"1-dimensional features: {len(dim_1_pairs)}")
    )pbdoc";
    
    // Module version and metadata
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "TDA Platform Team";
    
    // Bind core functionality
    bind_core(m);
    
    // Bind algorithms
    bind_algorithms(m);
    
    // Bind vector stack operations
    bind_vector_stack(m);
    
    // =============================================================================
    // Module-level convenience functions
    // =============================================================================
    
    // Quick access to common functionality
    m.def("create_point_cloud", [](py::array_t<double> points) {
        // For now, return a simple message since the full implementation isn't ready
        py::dict result;
        result["success"] = true;
        result["message"] = "Point cloud creation not yet implemented";
        result["num_points"] = points.shape(0);
        result["dimension"] = points.shape(1);
        return result;
    }, R"pbdoc(
        Create a point cloud from a NumPy array.
        
        Parameters
        ----------
        points : numpy.ndarray
            2D array of shape (n_points, n_dimensions)
            
        Returns
        -------
        dict
            Basic information about the input data
            
        Example
        -------
        >>> import numpy as np
        >>> points = np.random.random((50, 2))
        >>> pc = tda.create_point_cloud(points)
    )pbdoc", py::arg("points"));
    
    m.def("quick_vr_analysis", [](py::array_t<double> points, 
                                  double max_edge_length = 2.0, 
                                  int max_dimension = 2) {
        // For now, return a simple success message since the full implementation isn't ready
        py::dict result;
        result["success"] = true;
        result["message"] = "Vietoris-Rips analysis not yet implemented";
        result["num_points"] = points.shape(0);
        result["dimension"] = points.shape(1);
        result["max_edge_length"] = max_edge_length;
        result["max_dimension"] = max_dimension;
        return result;
    }, R"pbdoc(
        Quick Vietoris-Rips analysis with default parameters.
        
        Parameters
        ----------
        points : numpy.ndarray
            2D array of shape (n_points, n_dimensions)
        max_edge_length : float, optional
            Maximum edge length for complex construction (default: 2.0)
        max_dimension : int, optional
            Maximum homological dimension to compute (default: 2)
            
        Returns
        -------
        dict
            Basic information about the input data
            
        Example
        -------
        >>> import numpy as np
        >>> points = np.random.random((100, 2))
        >>> result = tda.quick_vr_analysis(points, max_edge_length=1.5)
        >>> print(f"Input has {result['num_points']} points")
    )pbdoc", py::arg("points"), py::arg("max_edge_length") = 2.0, py::arg("max_dimension") = 2);
    
    // Information functions
    m.def("version_info", []() {
        py::dict info;
        info["version"] = "1.0.0";
        info["build_type"] = "development";  // TODO: Set from CMake
        info["cpp_standard"] = "C++23";
        info["has_openmp"] = true;  // TODO: Detect at build time
        info["has_cuda"] = false;   // TODO: Detect at build time
        info["supported_algorithms"] = py::make_tuple(
            "vietoris_rips", "alpha_complex", "cech_complex", "dtm_filtration"
        );
        return info;
    }, "Get detailed version and build information");
    
    m.def("system_info", []() {
        py::dict info;
        info["simd_alignment"] = 32;  // SIMD_ALIGNMENT from types.hpp
        info["cache_line_size"] = 64; // CACHE_LINE_SIZE from types.hpp
        info["max_threads"] = std::thread::hardware_concurrency();
        return info;
    }, "Get system and performance information");
    
    // =============================================================================
    // Exception handling
    // =============================================================================
    
    // Custom exception types
    py::register_exception<std::invalid_argument>(m, "InvalidArgumentError");
    py::register_exception<std::runtime_error>(m, "ComputationError");
    py::register_exception<std::bad_alloc>(m, "MemoryError");
    
    // Exception translation
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, e.what());
        }
    });
}