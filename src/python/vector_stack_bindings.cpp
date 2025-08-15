#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "tda/core/types.hpp"
#include "tda/core/point_cloud.hpp"
// Future includes when vector stack is implemented:
// #include "tda/vector_stack/vector_stack.hpp"
// #include "tda/vector_stack/persistence_diagram.hpp"
// #include "tda/vector_stack/betti_numbers.hpp"
// #include "tda/vector_stack/vector_operations.hpp"

namespace py = pybind11;
using namespace tda::core;

// TODO: Create placeholder vector stack structures until C++ classes are implemented
struct VectorStackConfig {
    size_t initial_capacity = 1024;
    size_t growth_factor = 2;
    bool enable_compression = true;
    double compression_threshold = 0.8;
    size_t max_memory_mb = 4096;
};

struct VectorStackStats {
    size_t total_vectors = 0;
    size_t compressed_vectors = 0;
    double compression_ratio = 0.0;
    size_t memory_usage_mb = 0;
    double efficiency_score = 0.0;
};

struct PersistenceDiagram {
    std::vector<PersistencePair> pairs_;
    Dimension dimension_ = 0;
    double filtration_threshold_ = 0.0;
    
    explicit PersistenceDiagram(Dimension dim = 0) : dimension_(dim) {}
    
    void add_pair(const PersistencePair& pair) {
        pairs_.push_back(pair);
    }
    
    size_t size() const { return pairs_.size(); }
    bool empty() const { return pairs_.empty(); }
    
    const std::vector<PersistencePair>& pairs() const { return pairs_; }
    std::vector<PersistencePair>& pairs() { return pairs_; }
};

struct VectorStackOperation {
    enum Type {
        Push, Pop, Compress, Decompress, Merge, Split
    };
    
    Type operation_type = Push;
    size_t vector_id = 0;
    double timestamp = 0.0;
    size_t memory_delta = 0;
    bool success = false;
};

struct VectorStack {
    std::vector<std::vector<double>> vectors_;
    VectorStackConfig config_;
    VectorStackStats stats_;
    std::vector<VectorStackOperation> operation_history_;
    
    explicit VectorStack(const VectorStackConfig& config = VectorStackConfig{}) 
        : config_(config) {
        vectors_.reserve(config_.initial_capacity);
    }
    
    void push_vector(const std::vector<double>& vec) {
        vectors_.push_back(vec);
        stats_.total_vectors = vectors_.size();
        
        VectorStackOperation op;
        op.operation_type = VectorStackOperation::Push;
        op.vector_id = vectors_.size() - 1;
        op.success = true;
        operation_history_.push_back(op);
    }
    
    std::vector<double> pop_vector() {
        if (vectors_.empty()) {
            throw std::runtime_error("Cannot pop from empty vector stack");
        }
        
        auto result = std::move(vectors_.back());
        vectors_.pop_back();
        stats_.total_vectors = vectors_.size();
        
        VectorStackOperation op;
        op.operation_type = VectorStackOperation::Pop;
        op.success = true;
        operation_history_.push_back(op);
        
        return result;
    }
    
    size_t size() const { return vectors_.size(); }
    bool empty() const { return vectors_.empty(); }
    
    const VectorStackStats& get_stats() const { return stats_; }
    const VectorStackConfig& get_config() const { return config_; }
};

void bind_vector_stack(py::module_& m) {
    // Vector stack submodule
    auto vs_module = m.def_submodule("vector_stack", "High-performance vector stack operations for TDA");
    
    // =============================================================================
    // Vector Stack Configuration
    // =============================================================================
    py::class_<VectorStackConfig>(vs_module, "VectorStackConfig",
        "Configuration parameters for vector stack operations")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("initial_capacity", &VectorStackConfig::initial_capacity,
                      "Initial capacity for vector storage")
        .def_readwrite("growth_factor", &VectorStackConfig::growth_factor,
                      "Growth factor for capacity expansion")
        .def_readwrite("enable_compression", &VectorStackConfig::enable_compression,
                      "Enable vector compression for memory efficiency")
        .def_readwrite("compression_threshold", &VectorStackConfig::compression_threshold,
                      "Threshold for triggering compression (0.0-1.0)")
        .def_readwrite("max_memory_mb", &VectorStackConfig::max_memory_mb,
                      "Maximum memory usage in MB")
        .def("__repr__", [](const VectorStackConfig& c) {
            return "<VectorStackConfig(capacity=" + std::to_string(c.initial_capacity) +
                   ", compression=" + (c.enable_compression ? "enabled" : "disabled") + ")>";
        });
    
    // =============================================================================
    // Vector Stack Statistics
    // =============================================================================
    py::class_<VectorStackStats>(vs_module, "VectorStackStats",
        "Statistics and performance metrics for vector stack")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("total_vectors", &VectorStackStats::total_vectors,
                      "Total number of vectors in stack")
        .def_readwrite("compressed_vectors", &VectorStackStats::compressed_vectors,
                      "Number of compressed vectors")
        .def_readwrite("compression_ratio", &VectorStackStats::compression_ratio,
                      "Overall compression ratio achieved")
        .def_readwrite("memory_usage_mb", &VectorStackStats::memory_usage_mb,
                      "Current memory usage in MB")
        .def_readwrite("efficiency_score", &VectorStackStats::efficiency_score,
                      "Overall efficiency score (0.0-1.0)")
        .def("__repr__", [](const VectorStackStats& s) {
            return "<VectorStackStats(vectors=" + std::to_string(s.total_vectors) +
                   ", compression=" + std::to_string(s.compression_ratio) + ")>";
        });
    
    // =============================================================================
    // Persistence Diagram
    // =============================================================================
    py::class_<PersistenceDiagram>(vs_module, "PersistenceDiagram",
        "Persistence diagram for visualizing topological features")
        .def(py::init<>(), "Default constructor")
        .def(py::init<Dimension>(), "Constructor with dimension", py::arg("dimension"))
        .def("add_pair", &PersistenceDiagram::add_pair,
             "Add a persistence pair to the diagram", py::arg("pair"))
        .def("size", &PersistenceDiagram::size, "Get number of persistence pairs")
        .def("empty", &PersistenceDiagram::empty, "Check if diagram is empty")
        .def("pairs", static_cast<const std::vector<PersistencePair>& (PersistenceDiagram::*)() const>(&PersistenceDiagram::pairs),
             "Get all persistence pairs", py::return_value_policy::reference_internal)
        .def_readwrite("dimension", &PersistenceDiagram::dimension_, "Homological dimension")
        .def_readwrite("filtration_threshold", &PersistenceDiagram::filtration_threshold_, 
                      "Filtration threshold used")
        .def("__len__", &PersistenceDiagram::size)
        .def("__repr__", [](const PersistenceDiagram& d) {
            return "<PersistenceDiagram(dim=" + std::to_string(d.dimension_) +
                   ", pairs=" + std::to_string(d.size()) + ")>";
        })
        .def("filter_by_persistence", [](const PersistenceDiagram& d, double min_persistence) {
            PersistenceDiagram result(d.dimension_);
            for (const auto& pair : d.pairs()) {
                if (pair.get_persistence() >= min_persistence) {
                    result.add_pair(pair);
                }
            }
            return result;
        }, "Filter pairs by minimum persistence", py::arg("min_persistence"))
        .def("to_numpy", [](const PersistenceDiagram& d) {
            if (d.empty()) {
                return py::array_t<double>();
            }
            
            // Create array with proper shape using the correct pybind11 API
            std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(d.size()), 2};
            auto result = py::array_t<double>(shape);
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            
            for (size_t i = 0; i < d.size(); ++i) {
                const auto& pair = d.pairs()[i];
                ptr[i * 2] = pair.birth;
                ptr[i * 2 + 1] = pair.death;
            }
            
            return result;
        }, "Convert to NumPy array (birth, death) pairs");
    
    // =============================================================================
    // Vector Stack Operation
    // =============================================================================
    py::enum_<VectorStackOperation::Type>(vs_module, "OperationType", "Types of vector stack operations")
        .value("Push", VectorStackOperation::Type::Push, "Push vector onto stack")
        .value("Pop", VectorStackOperation::Type::Pop, "Pop vector from stack")
        .value("Compress", VectorStackOperation::Type::Compress, "Compress vectors")
        .value("Decompress", VectorStackOperation::Type::Decompress, "Decompress vectors")
        .value("Merge", VectorStackOperation::Type::Merge, "Merge vector stacks")
        .value("Split", VectorStackOperation::Type::Split, "Split vector stack")
        .export_values();
    
    py::class_<VectorStackOperation>(vs_module, "VectorStackOperation",
        "Record of a vector stack operation")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("operation_type", &VectorStackOperation::operation_type, "Type of operation")
        .def_readwrite("vector_id", &VectorStackOperation::vector_id, "Vector ID involved")
        .def_readwrite("timestamp", &VectorStackOperation::timestamp, "Operation timestamp")
        .def_readwrite("memory_delta", &VectorStackOperation::memory_delta, "Memory change in bytes")
        .def_readwrite("success", &VectorStackOperation::success, "Whether operation succeeded")
        .def("__repr__", [](const VectorStackOperation& op) {
            return "<VectorStackOperation(type=" + std::to_string(static_cast<int>(op.operation_type)) +
                   ", success=" + (op.success ? "true" : "false") + ")>";
        });
    
    // =============================================================================
    // Vector Stack Main Class
    // =============================================================================
    py::class_<VectorStack>(vs_module, "VectorStack",
        "High-performance vector stack for TDA computations")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const VectorStackConfig&>(), "Constructor with configuration", py::arg("config"))
        .def("push_vector", &VectorStack::push_vector,
             "Push a vector onto the stack", py::arg("vector"))
        .def("pop_vector", &VectorStack::pop_vector,
             "Pop a vector from the stack")
        .def("size", &VectorStack::size, "Get number of vectors in stack")
        .def("empty", &VectorStack::empty, "Check if stack is empty")
        .def("get_stats", &VectorStack::get_stats,
             "Get performance statistics", py::return_value_policy::reference_internal)
        .def("get_config", &VectorStack::get_config,
             "Get configuration", py::return_value_policy::reference_internal)
        .def_readonly("operation_history", &VectorStack::operation_history_,
                     "History of operations performed")
        .def("__len__", &VectorStack::size)
        .def("__repr__", [](const VectorStack& vs) {
            return "<VectorStack(size=" + std::to_string(vs.size()) + ")>";
        })
        .def("push_from_numpy", [](VectorStack& vs, py::array_t<double> array) {
            auto buf = array.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Array must be 1-dimensional");
            }
            
            std::vector<double> vec(buf.shape[0]);
            double* ptr = static_cast<double*>(buf.ptr);
            for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
                vec[i] = ptr[i];
            }
            
            vs.push_vector(vec);
        }, "Push vector from NumPy array", py::arg("array"))
        .def("pop_to_numpy", [](VectorStack& vs) {
            auto vec = vs.pop_vector();
            auto result = py::array_t<double>(vec.size());
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            
            for (size_t i = 0; i < vec.size(); ++i) {
                ptr[i] = vec[i];
            }
            
            return result;
        }, "Pop vector as NumPy array");
    
    // =============================================================================
    // Utility Functions
    // =============================================================================
    vs_module.def("create_default_config", []() {
        VectorStackConfig config;
        config.initial_capacity = 1024;
        config.growth_factor = 2;
        config.enable_compression = true;
        config.compression_threshold = 0.8;
        config.max_memory_mb = 4096;
        return config;
    }, "Create default vector stack configuration");
    
    vs_module.def("create_performance_config", []() {
        VectorStackConfig config;
        config.initial_capacity = 8192;
        config.growth_factor = 3;
        config.enable_compression = false;  // Disable for speed
        config.max_memory_mb = 16384;       // Allow more memory
        return config;
    }, "Create high-performance configuration");
    
    vs_module.def("create_memory_efficient_config", []() {
        VectorStackConfig config;
        config.initial_capacity = 256;
        config.growth_factor = 1;
        config.enable_compression = true;
        config.compression_threshold = 0.5;  // Aggressive compression
        config.max_memory_mb = 1024;         // Limited memory
        return config;
    }, "Create memory-efficient configuration");
    
    // Persistence diagram utilities
    vs_module.def("merge_diagrams", [](const std::vector<PersistenceDiagram>& diagrams) {
        if (diagrams.empty()) {
            return PersistenceDiagram();
        }
        
        PersistenceDiagram result(diagrams[0].dimension_);
        for (const auto& diagram : diagrams) {
            for (const auto& pair : diagram.pairs()) {
                result.add_pair(pair);
            }
        }
        
        return result;
    }, "Merge multiple persistence diagrams", py::arg("diagrams"));
    
    vs_module.def("compute_diagram_distance", [](const PersistenceDiagram& d1, const PersistenceDiagram& d2) {
        // TODO: Implement proper persistence diagram distance (e.g., Wasserstein, bottleneck)
        // For now, return a simple metric based on size difference
        double size_diff = std::abs(static_cast<double>(d1.size()) - static_cast<double>(d2.size()));
        return size_diff / std::max(d1.size(), d2.size());
    }, "Compute distance between persistence diagrams", py::arg("diagram1"), py::arg("diagram2"));
    
    // Vector stack manipulation
    vs_module.def("create_stack_from_point_cloud", [](const PointCloud& pc, const VectorStackConfig& config) {
        VectorStack stack(config);
        
        // Add each point as a vector to the stack
        for (const auto& point : pc.points()) {
            stack.push_vector(point);
        }
        
        return stack;
    }, "Create vector stack from point cloud", py::arg("point_cloud"), py::arg("config") = VectorStackConfig{});
    
    vs_module.def("benchmark_stack_operations", [](size_t num_operations, size_t vector_size) {
        VectorStackConfig config;
        config.initial_capacity = num_operations;
        VectorStack stack(config);
        
        std::vector<double> test_vector(vector_size, 1.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Push operations
        for (size_t i = 0; i < num_operations; ++i) {
            stack.push_vector(test_vector);
        }
        
        // Pop operations
        for (size_t i = 0; i < num_operations; ++i) {
            stack.pop_vector();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        py::dict result;
        result["total_time_us"] = duration.count();
        result["operations_per_second"] = (2.0 * num_operations * 1000000.0) / duration.count();
        result["average_time_per_op_us"] = static_cast<double>(duration.count()) / (2.0 * num_operations);
        
        return result;
    }, "Benchmark vector stack operations", py::arg("num_operations") = 10000, py::arg("vector_size") = 100);
}
