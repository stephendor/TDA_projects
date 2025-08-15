# Bilingual Development Guide: Python ↔ C++23 for TDA Vector Stack

## 🎯 **Overview**

This guide demonstrates how to translate between Python and C++23 implementations in the TDA Vector Stack project. We maintain both languages to leverage Python's expressiveness for API design and C++23's performance for core computations.

## 🔄 **Translation Patterns**

### **1. TDA Computation Engine**

#### **Python API Design (Reference)**
```python
class TDAComputationRequest(BaseModel):
    point_cloud: List[List[float]]
    filtration_type: str
    max_dimension: int = 3
    parameters: Optional[dict] = None

class TDAComputationResponse(BaseModel):
    job_id: str
    status: str
    persistence_diagram: Optional[dict] = None
    betti_numbers: Optional[dict] = None

@app.post("/tda/compute", response_model=TDAComputationResponse)
async def compute_persistence(request: TDAComputationRequest):
    try:
        # Validate input
        if len(request.point_cloud) > 1_000_000:
            raise HTTPException(status_code=400, detail="Point cloud too large")
        
        # Process computation
        result = await tda_engine.compute(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### **C++23 Core Implementation**
```cpp
// ✅ DO: Use modern C++23 features for maximum performance and safety
#include <memory>
#include <expected>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace tda::core {

// Request structure matching Python API
struct TDAComputationRequest {
    std::vector<std::vector<double>> point_cloud;
    std::string filtration_type;
    int max_dimension = 3;
    std::optional<std::unordered_map<std::string, double>> parameters;
};

// Response structure matching Python API
struct TDAComputationResponse {
    std::string job_id;
    std::string status;
    std::optional<PersistenceDiagram> persistence_diagram;
    std::optional<BettiNumbers> betti_numbers;
};

// Core computation engine using C++23 features
class PersistentHomologyEngine {
public:
    // Use std::expected for robust error handling without exceptions
    std::expected<PersistenceDiagram, ComputationError> computePersistence(
        std::span<const std::vector<double>> point_cloud, 
        FiltrationType type,
        int max_dimension = 3
    ) {
        // Validate input using C++23 features
        if (point_cloud.size() > 1'000'000) {
            return std::unexpected(ComputationError::PointCloudTooLarge);
        }
        
        // Use ranges for functional-style processing
        auto valid_points = point_cloud | std::views::filter([](const auto& point) {
            return point.size() >= 2 && std::ranges::all_of(point, [](double d) {
                return std::isfinite(d);
            });
        });
        
        // Process computation
        return computePersistenceInternal(valid_points, type, max_dimension);
    }
    
private:
    std::expected<PersistenceDiagram, ComputationError> computePersistenceInternal(
        auto valid_points, FiltrationType type, int max_dimension);
};

} // namespace tda::core
```

### **2. Vector Stack Implementation**

#### **Python Vectorization (Reference)**
```python
class VectorStack:
    def __init__(self, config: VectorStackConfig):
        self.config = config
        self.blocks = self._initialize_blocks()
    
    def _initialize_blocks(self):
        blocks = []
        
        if self.config.enable_persistence_images:
            blocks.append(PersistenceImageBlock(
                resolution=self.config.image_resolution,
                intensity_normalize=self.config.intensity_normalize_images,
                use_log_lifetime=self.config.use_log_lifetime_images
            ))
        
        if self.config.enable_landscapes:
            blocks.append(PersistenceLandscapeBlock(
                resolution=self.config.landscape_resolution,
                global_range=self.config.global_landscape_range
            ))
        
        return blocks
    
    def vectorize(self, persistence_diagram: PersistenceDiagram) -> np.ndarray:
        features = []
        for block in self.blocks:
            block_features = block.compute(persistence_diagram)
            features.append(block_features)
        
        return np.concatenate(features, axis=0)
```

#### **C++23 Vectorization Implementation**
```cpp
namespace tda::vector_stack {

// Configuration structure matching Python config
struct VectorStackConfig {
    bool enable_persistence_images = true;
    bool enable_landscapes = true;
    bool enable_sliced_wasserstein = true;
    
    int image_resolution = 100;
    int landscape_resolution = 100;
    int sw_num_angles = 100;
    
    bool intensity_normalize_images = true;
    bool use_log_lifetime_images = false;
    std::pair<double, double> global_landscape_range = {0.0, 1.0};
};

// Base block interface using C++23 concepts
template<typename T>
concept VectorizationBlock = requires(T t, const PersistenceDiagram& pd) {
    { t.compute(pd) } -> std::convertible_to<std::vector<double>>;
    { t.name() } -> std::convertible_to<std::string>;
};

// Vector stack implementation using modern C++23
class VectorStack {
public:
    explicit VectorStack(const VectorStackConfig& config) 
        : config_(config) {
        initializeBlocks();
    }
    
    // Use std::expected for error handling
    std::expected<std::vector<double>, VectorizationError> vectorize(
        const PersistenceDiagram& persistence_diagram) {
        
        std::vector<double> features;
        
        // Use ranges for functional composition
        for (const auto& block : blocks_) {
            auto block_features = block->compute(persistence_diagram);
            if (!block_features.has_value()) {
                return std::unexpected(block_features.error());
            }
            
            // Efficient concatenation using ranges
            auto feature_span = std::span(block_features.value());
            features.insert(features.end(), feature_span.begin(), feature_span.end());
        }
        
        return features;
    }
    
private:
    void initializeBlocks() {
        if (config_.enable_persistence_images) {
            blocks_.push_back(std::make_unique<PersistenceImageBlock>(
                config_.image_resolution,
                config_.intensity_normalize_images,
                config_.use_log_lifetime_images
            ));
        }
        
        if (config_.enable_landscapes) {
            blocks_.push_back(std::make_unique<PersistenceLandscapeBlock>(
                config_.landscape_resolution,
                config_.global_landscape_range
            ));
        }
        
        if (config_.enable_sliced_wasserstein) {
            blocks_.push_back(std::make_unique<SlicedWassersteinBlock>(
                config_.sw_num_angles,
                AngleStrategy::Halton,
                config_.sw_resolution
            ));
        }
    }
    
    VectorStackConfig config_;
    std::vector<std::unique_ptr<VectorizationBlock>> blocks_;
};

} // namespace tda::vector_stack
```

### **3. Deep Learning Integration**

#### **Python PyTorch Layer (Reference)**
```python
class PersistenceAttentionLayer(TDALayer):
    """Learnable attention over persistence diagrams"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__(feature_dim, feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features, feature_dim)
        attended, _ = self.attention(x, x, x)
        return attended
```

#### **C++23 Implementation with pybind11**
```cpp
namespace tda::models {

// C++23 implementation of attention mechanism
class PersistenceAttentionLayer {
public:
    PersistenceAttentionLayer(int feature_dim, int num_heads = 8)
        : feature_dim_(feature_dim), num_heads_(num_heads) {
        initializeAttention();
    }
    
    // Use std::expected for error handling
    std::expected<std::vector<std::vector<double>>, AttentionError> forward(
        const std::vector<std::vector<double>>& input) {
        
        // Validate input dimensions
        if (input.empty() || input[0].size() != feature_dim_) {
            return std::unexpected(AttentionError::InvalidInputDimensions);
        }
        
        // Compute attention using C++23 features
        auto attention_scores = computeAttentionScores(input);
        if (!attention_scores.has_value()) {
            return std::unexpected(attention_scores.error());
        }
        
        // Apply attention weights
        return applyAttention(input, attention_scores.value());
    }
    
private:
    void initializeAttention() {
        // Initialize attention weights using modern C++23
        attention_weights_ = std::make_unique<AttentionWeights>(feature_dim_, num_heads_);
    }
    
    std::expected<std::vector<std::vector<double>>, AttentionError> computeAttentionScores(
        const std::vector<std::vector<double>>& input);
    
    std::vector<std::vector<double>> applyAttention(
        const std::vector<std::vector<double>>& input,
        const std::vector<std::vector<double>>& scores);
    
    int feature_dim_;
    int num_heads_;
    std::unique_ptr<AttentionWeights> attention_weights_;
};

} // namespace tda::models

// pybind11 bindings to expose C++ to Python
PYBIND11_MODULE(tda_models, m) {
    py::class_<tda::models::PersistenceAttentionLayer>(m, "PersistenceAttentionLayer")
        .def(py::init<int, int>(), py::arg("feature_dim"), py::arg("num_heads") = 8)
        .def("forward", &tda::models::PersistenceAttentionLayer::forward)
        .def_property_readonly("feature_dim", &tda::models::PersistenceAttentionLayer::feature_dim_)
        .def_property_readonly("num_heads", &tda::models::PersistenceAttentionLayer::num_heads_);
}
```

## 🔄 **Translation Guidelines**

### **1. Data Structures**
- **Python**: Use Pydantic models for API validation
- **C++23**: Use structs with std::optional for optional fields
- **Translation**: Keep field names identical for easy mapping

### **2. Error Handling**
- **Python**: Use exceptions and HTTP status codes
- **C++23**: Use std::expected for computation errors
- **Translation**: Map C++ error codes to Python exceptions in bindings

### **3. Collections and Iteration**
- **Python**: Use list comprehensions and NumPy arrays
- **C++23**: Use std::vector with std::ranges and std::span
- **Translation**: Use ranges for functional-style operations

### **4. Configuration**
- **Python**: Use dataclasses or Pydantic models
- **C++23**: Use structs with default values
- **Translation**: Keep parameter names identical

### **5. Performance Critical Paths**
- **Python**: Use for orchestration and API
- **C++23**: Use for mathematical computations and data processing
- **Translation**: Python calls C++ through pybind11 bindings

## 🧪 **Testing Bilingual Code**

### **1. Unit Tests in Both Languages**
```cpp
// C++23 test
TEST_CASE("VectorStack vectorization") {
    VectorStackConfig config;
    config.enable_persistence_images = true;
    config.enable_landscapes = true;
    
    VectorStack stack(config);
    PersistenceDiagram diagram = createTestDiagram();
    
    auto result = stack.vectorize(diagram);
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() > 0);
}
```

```python
# Python test
def test_vector_stack_vectorization():
    config = VectorStackConfig(
        enable_persistence_images=True,
        enable_landscapes=True
    )
    
    stack = VectorStack(config)
    diagram = create_test_diagram()
    
    result = stack.vectorize(diagram)
    assert result is not None
    assert len(result) > 0
```

### **2. Integration Tests**
```python
def test_cpp_python_integration():
    # Test that C++ implementation matches Python
    cpp_result = cpp_vector_stack.vectorize(diagram)
    python_result = python_vector_stack.vectorize(diagram)
    
    np.testing.assert_array_almost_equal(cpp_result, python_result)
```

## 🎯 **Best Practices**

### **1. Keep APIs Identical**
- Field names should match between Python and C++
- Method signatures should be similar
- Error handling should be consistent

### **2. Use C++23 for Performance**
- Mathematical computations
- Data processing pipelines
- Memory-intensive operations
- SIMD-optimized algorithms

### **3. Use Python for Orchestration**
- API endpoints
- Configuration management
- High-level workflow logic
- Integration with ML frameworks

### **4. Maintain Both Implementations**
- Update both when changing algorithms
- Test both implementations
- Document differences and trade-offs
- Use Python as the source of truth for API design

## 📚 **Examples Repository**

### **1. Core TDA Algorithms**
- **Python**: API design and high-level logic
- **C++23**: Performance-critical implementations
- **Tests**: Both languages with integration tests

### **2. Vector Stack Features**
- **Python**: Configuration and orchestration
- **C++23**: Mathematical computations
- **Bindings**: pybind11 for seamless integration

### **3. Deep Learning Integration**
- **Python**: PyTorch layer definitions
- **C++23**: Core attention mechanisms
- **Performance**: C++ for training, Python for inference

---

## 🎯 **Next Steps**

1. **Implement C++23 versions** of all Python examples
2. **Create pybind11 bindings** for seamless integration
3. **Write bilingual tests** to ensure consistency
4. **Document performance differences** between implementations
5. **Create migration guides** for team members

---

*This guide ensures we maintain the best of both worlds: Python's expressiveness for API design and C++23's performance for core computations.*
