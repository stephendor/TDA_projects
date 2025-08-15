#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "tda/core/types.hpp"
#include "tda/core/point_cloud.hpp"
// #include "tda/core/simplex.hpp"
// #include "tda/core/filtration.hpp"
// #include "tda/core/persistent_homology.hpp"

namespace py = pybind11;
using namespace tda::core;

void bind_core(py::module_& m) {
    // Core types submodule
    auto types_module = m.def_submodule("types", "Core TDA types and utilities");
    
    // =============================================================================
    // Basic Type Aliases and Constants
    // =============================================================================
    types_module.attr("SIMD_ALIGNMENT") = SIMD_ALIGNMENT;
    types_module.attr("CACHE_LINE_SIZE") = CACHE_LINE_SIZE;
    
    // =============================================================================
    // Error Code Enumeration
    // =============================================================================
    py::enum_<ErrorCode>(types_module, "ErrorCode", "Error codes for TDA operations")
        .value("Success", ErrorCode::Success, "Operation completed successfully")
        .value("InvalidInput", ErrorCode::InvalidInput, "Invalid input parameters")
        .value("MemoryAllocationFailed", ErrorCode::MemoryAllocationFailed, "Memory allocation failed")
        .value("ComputationFailed", ErrorCode::ComputationFailed, "Computation failed")
        .value("NotImplemented", ErrorCode::NotImplemented, "Feature not yet implemented")
        .export_values();
    
    // =============================================================================
    // PersistencePair Structure
    // =============================================================================
    py::class_<PersistencePair>(types_module, "PersistencePair", 
        "Represents a persistence pair (birth, death) with associated dimension")
        .def(py::init<>(), "Default constructor")
        .def(py::init<Birth, Death, Dimension, Index, Index>(),
             "Full constructor",
             py::arg("birth"), py::arg("death"), py::arg("dimension"),
             py::arg("birth_simplex") = 0, py::arg("death_simplex") = 0)
        .def(py::init<Dimension, Birth, Death>(),
             "Simple constructor for TDA algorithms",
             py::arg("dimension"), py::arg("birth"), py::arg("death"))
        .def_readwrite("birth", &PersistencePair::birth, "Birth time of the feature")
        .def_readwrite("death", &PersistencePair::death, "Death time of the feature")
        .def_readwrite("dimension", &PersistencePair::dimension, "Homological dimension")
        .def_readwrite("birth_simplex", &PersistencePair::birth_simplex, "Birth simplex index")
        .def_readwrite("death_simplex", &PersistencePair::death_simplex, "Death simplex index")
        .def("get_persistence", &PersistencePair::get_persistence,
             "Get persistence (death - birth)")
        .def("is_finite", &PersistencePair::is_finite,
             "Check if death time is finite")
        .def("is_infinite", &PersistencePair::is_infinite,
             "Check if death time is infinite")
        .def("__repr__", [](const PersistencePair& p) {
            return "<PersistencePair(dim=" + std::to_string(p.dimension) +
                   ", birth=" + std::to_string(p.birth) +
                   ", death=" + std::to_string(p.death) + ")>";
        })
        .def("__str__", [](const PersistencePair& p) {
            return "(" + std::to_string(p.birth) + ", " + std::to_string(p.death) + ")";
        });
    
    // =============================================================================
    // BettiNumbers Structure
    // =============================================================================
    py::class_<BettiNumbers>(types_module, "BettiNumbers",
        "Betti numbers for different homological dimensions")
        .def(py::init<>(), "Default constructor")
        .def(py::init<Dimension>(), "Constructor with maximum dimension", py::arg("max_dim"))
        .def("__getitem__", [](const BettiNumbers& b, Dimension dim) { return b[dim]; },
             "Get Betti number for dimension")
        .def("__setitem__", [](BettiNumbers& b, Dimension dim, Index value) { b[dim] = value; },
             "Set Betti number for dimension")
        .def("max_dimension", &BettiNumbers::max_dimension,
             "Get maximum dimension")
        .def("total_betti", &BettiNumbers::total_betti,
             "Get sum of all Betti numbers")
        .def("__len__", [](const BettiNumbers& b) { return b.max_dimension() + 1; })
        .def("__repr__", [](const BettiNumbers& b) {
            std::string result = "<BettiNumbers([";
            for (Dimension i = 0; i <= b.max_dimension(); ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(b[i]);
            }
            result += "])>";
            return result;
        })
        .def("to_list", [](const BettiNumbers& b) {
            std::vector<Index> result;
            for (Dimension i = 0; i <= b.max_dimension(); ++i) {
                result.push_back(b[i]);
            }
            return result;
        }, "Convert to Python list");
    
    // =============================================================================
    // SimplexInfo Structure
    // =============================================================================
    py::class_<SimplexInfo>(types_module, "SimplexInfo",
        "Information about a simplex in a simplicial complex")
        .def(py::init<>(), "Default constructor")
        .def(py::init<int, double, std::vector<int>>(),
             "Constructor with dimension, filtration value, and vertices",
             py::arg("dimension"), py::arg("filtration_value"), py::arg("vertices"))
        .def_readwrite("dimension", &SimplexInfo::dimension, "Simplex dimension")
        .def_readwrite("filtration_value", &SimplexInfo::filtration_value, "Filtration value")
        .def_readwrite("vertices", &SimplexInfo::vertices, "Vertex indices")
        .def("__repr__", [](const SimplexInfo& s) {
            return "<SimplexInfo(dim=" + std::to_string(s.dimension) +
                   ", filt=" + std::to_string(s.filtration_value) + ")>";
        });
    
    // =============================================================================
    // ComplexStatistics Structure
    // =============================================================================
    py::class_<ComplexStatistics>(types_module, "ComplexStatistics",
        "Statistics about a simplicial complex")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("num_points", &ComplexStatistics::num_points, "Number of points")
        .def_readwrite("num_simplices", &ComplexStatistics::num_simplices, "Number of simplices")
        .def_readwrite("max_dimension", &ComplexStatistics::max_dimension, "Maximum dimension")
        .def_readwrite("threshold", &ComplexStatistics::threshold, "Filtration threshold")
        .def_readwrite("simplex_count_by_dim", &ComplexStatistics::simplex_count_by_dim,
                      "Simplex count by dimension")
        .def("__repr__", [](const ComplexStatistics& s) {
            return "<ComplexStatistics(points=" + std::to_string(s.num_points) +
                   ", simplices=" + std::to_string(s.num_simplices) +
                   ", max_dim=" + std::to_string(s.max_dimension) + ")>";
        });
    
    // =============================================================================
    // Result Template (for error handling)
    // =============================================================================
    
    // Result for vector of PersistencePairs
    py::class_<Result<std::vector<PersistencePair>>>(types_module, "PersistenceResult",
        "Result containing persistence pairs or error message")
        .def("has_value", &Result<std::vector<PersistencePair>>::has_value,
             "Check if result contains a value")
        .def("has_error", &Result<std::vector<PersistencePair>>::has_error,
             "Check if result contains an error")
        .def("value", static_cast<const std::vector<PersistencePair>& (Result<std::vector<PersistencePair>>::*)() const>(&Result<std::vector<PersistencePair>>::value),
             "Get the result value (throws if error)",
             py::return_value_policy::reference_internal)
        .def("error", &Result<std::vector<PersistencePair>>::error,
             "Get the error message")
        .def("__bool__", &Result<std::vector<PersistencePair>>::has_value,
             "Check if result is successful")
        .def("__repr__", [](const Result<std::vector<PersistencePair>>& r) {
            if (r.has_value()) {
                return "<PersistenceResult: " + std::to_string(r.value().size()) + " pairs>";
            } else {
                return "<PersistenceResult: Error - " + r.error() + ">";
            }
        });
    
    // Result for BettiNumbers
    py::class_<Result<BettiNumbers>>(types_module, "BettiResult",
        "Result containing Betti numbers or error message")
        .def("has_value", &Result<BettiNumbers>::has_value)
        .def("has_error", &Result<BettiNumbers>::has_error)
        .def("value", static_cast<const BettiNumbers& (Result<BettiNumbers>::*)() const>(&Result<BettiNumbers>::value),
             py::return_value_policy::reference_internal)
        .def("error", &Result<BettiNumbers>::error)
        .def("__bool__", &Result<BettiNumbers>::has_value);
    
    // Result for void (operation success/failure)
    py::class_<Result<void>>(types_module, "OperationResult",
        "Result indicating operation success or failure")
        .def("has_value", &Result<void>::has_value)
        .def("has_error", &Result<void>::has_error)
        .def("error", &Result<void>::error)
        .def("__bool__", &Result<void>::has_value)
        .def("__repr__", [](const Result<void>& r) {
            return r.has_value() ? "<OperationResult: Success>" 
                                 : "<OperationResult: Error - " + r.error() + ">";
        });
    
    // =============================================================================
    // PointCloud Class
    // =============================================================================
    py::class_<PointCloud>(m, "PointCloud", 
        "Point cloud data structure for TDA computations")
        .def(py::init<>(), "Default constructor")
        .def(py::init<PointCloud::PointContainer>(),
             "Constructor from point container",
             py::arg("points"))
        .def("__getitem__", [](const PointCloud& pc, size_t index) -> const PointCloud::Point& {
            if (index >= pc.size()) {
                throw py::index_error("Point index out of range");
            }
            return pc[index];
        }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](PointCloud& pc, size_t index, const PointCloud::Point& point) {
            if (index >= pc.size()) {
                throw py::index_error("Point index out of range");
            }
            pc[index] = point;
        })
        .def("__len__", &PointCloud::size)
        .def("size", &PointCloud::size, "Get number of points")
        .def("dimension", &PointCloud::dimension, "Get point dimension")
        .def("empty", &PointCloud::empty, "Check if point cloud is empty")
        .def("is_valid", &PointCloud::isValid, "Check if point cloud is valid")
        .def("add_point", 
             py::overload_cast<const PointCloud::Point&>(&PointCloud::addPoint),
             "Add a point to the cloud",
             py::arg("point"))
        .def("clear", &PointCloud::clear, "Clear all points")
        .def("reserve", &PointCloud::reserve, "Reserve capacity for points", py::arg("capacity"))
        .def("points", [](const PointCloud& pc) { return pc.points(); },
             "Get all points as a list",
             py::return_value_policy::reference_internal)
        
        // NumPy integration
        .def("to_numpy", [](const PointCloud& pc) {
            if (pc.empty()) {
                return py::array_t<double>();
            }
            
            auto points = pc.points();
            size_t n_points = points.size();
            size_t n_dims = points[0].size();
            
            auto result = py::array_t<double>({n_points, n_dims});
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            
            for (size_t i = 0; i < n_points; ++i) {
                for (size_t j = 0; j < n_dims; ++j) {
                    ptr[i * n_dims + j] = points[i][j];
                }
            }
            
            return result;
        }, "Convert to NumPy array")
        
        .def_static("from_numpy", [](py::array_t<double> array) {
            auto buf = array.request();
            
            if (buf.ndim != 2) {
                throw std::runtime_error("Array must be 2-dimensional");
            }
            
            size_t n_points = buf.shape[0];
            size_t n_dims = buf.shape[1];
            double* ptr = static_cast<double*>(buf.ptr);
            
            PointCloud::PointContainer points;
            points.reserve(n_points);
            
            for (size_t i = 0; i < n_points; ++i) {
                PointCloud::Point point(n_dims);
                for (size_t j = 0; j < n_dims; ++j) {
                    point[j] = ptr[i * n_dims + j];
                }
                points.push_back(std::move(point));
            }
            
            return PointCloud(std::move(points));
        }, "Create from NumPy array", py::arg("array"))
        
        .def("__repr__", [](const PointCloud& pc) {
            return "<PointCloud: " + std::to_string(pc.size()) + " points, " +
                   std::to_string(pc.dimension()) + "D>";
        })
        .def("__iter__", [](const PointCloud& pc) {
            return py::make_iterator(pc.begin(), pc.end());
        }, py::keep_alive<0, 1>());
    
    // =============================================================================
    // Utility Functions
    // =============================================================================
    types_module.def("is_power_of_two", [](uint64_t n) { return is_power_of_two(n); },
                     "Check if number is power of two", py::arg("n"));
    
    types_module.def("next_power_of_two", [](uint64_t n) { return next_power_of_two(n); },
                     "Get next power of two", py::arg("n"));
    
    types_module.def("aligned_size", [](size_t count, size_t element_size) {
        return ((count * element_size + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;
    }, "Get SIMD-aligned size", py::arg("count"), py::arg("element_size"));
    
    // =============================================================================
    // Factory Functions for Results
    // =============================================================================
    types_module.def("success", [](std::vector<PersistencePair> pairs) {
        return Result<std::vector<PersistencePair>>::success(std::move(pairs));
    }, "Create successful persistence result", py::arg("pairs"));
    
    types_module.def("failure", [](const std::string& error) {
        return Result<std::vector<PersistencePair>>::failure(error);
    }, "Create failed persistence result", py::arg("error"));
    
    types_module.def("betti_success", [](BettiNumbers betti) {
        return Result<BettiNumbers>::success(std::move(betti));
    }, "Create successful Betti result", py::arg("betti"));
    
    types_module.def("betti_failure", [](const std::string& error) {
        return Result<BettiNumbers>::failure(error);
    }, "Create failed Betti result", py::arg("error"));
    
    types_module.def("operation_success", []() {
        return Result<void>::success();
    }, "Create successful operation result");
    
    types_module.def("operation_failure", [](const std::string& error) {
        return Result<void>::failure(error);
    }, "Create failed operation result", py::arg("error"));
}