# TDA Platform - Project-Wide BugBot Rules

## Project Overview
This is a **Topological Data Analysis (TDA) platform** that combines C++23 performance-critical algorithms with Python backend services and React frontend. The platform provides persistent homology computation, Mapper algorithm implementation, and specialized modules for finance and cybersecurity applications.

## Architecture & Technology Stack

### Core Components
- **C++23 Core Engine**: High-performance TDA algorithms (Vietoris-Rips, Alpha Complex, Čech, DTM filtrations)
- **Python Backend**: FastAPI services with C++ bindings via pybind11
- **React Frontend**: Modern UI with D3.js/Three.js visualizations
- **Hybrid Database**: PostgreSQL (metadata) + MongoDB (diagrams/vectors)
- **Streaming Pipeline**: Apache Kafka + Apache Flink for real-time processing

### Key Algorithms
- Persistent homology computation (dimensions 0-3)
- Mapper algorithm with customizable filter functions
- Distance-to-Measure (DTM) filtrations
- Learnable topological features (PyTorch integration)

## Project-Specific Rules

### 1. C++ Code Standards
- **C++23 Standard**: Use modern C++ features (concepts, ranges, modules)
- **Performance Critical**: Core TDA algorithms must handle >1M points in <60 seconds
- **Memory Management**: Use RAII, smart pointers, avoid raw pointers
- **SIMD Optimization**: Leverage vectorization for mathematical operations
- **Exception Safety**: Provide strong exception guarantees

### 2. Python Backend Rules
- **FastAPI Best Practices**: Use dependency injection, proper error handling
- **Type Hints**: Mandatory for all function signatures
- **Async/Await**: Use for I/O operations, database queries
- **C++ Integration**: Ensure proper pybind11 binding patterns
- **Database Transactions**: Use proper transaction management for dual-database operations

### 3. Frontend Development
- **React 18+**: Use hooks, functional components, modern patterns
- **TypeScript**: Strict mode, proper type definitions
- **Visualization Libraries**: D3.js for 2D, Three.js for 3D, Plotly for interactive charts
- **Responsive Design**: Mobile-first approach, accessibility compliance (WCAG 2.1 AA)
- **State Management**: Use React Query for server state, Context for UI state

### 4. Database Design
- **PostgreSQL**: Metadata, versioning, audit trails, user management
- **MongoDB**: Raw persistence diagrams, feature vectors, large binary data
- **Data Consistency**: Implement eventual consistency patterns
- **Audit Trail**: Immutable logging for compliance (ST-201, ST-202)

### 5. Security Requirements
- **Authentication**: JWT-based with MFA support
- **RBAC**: Role-based access control for all endpoints
- **Data Encryption**: At-rest (TDE) and in-transit (TLS)
- **Audit Logging**: Comprehensive event tracking for compliance

### 6. Performance Requirements
- **Scalability**: Handle enterprise workloads (10K+ events/sec)
- **Distributed Processing**: Spark/Flink integration for large datasets
- **GPU Acceleration**: CUDA optimization for compute-intensive operations
- **Horizontal Scaling**: Kubernetes deployment with auto-scaling

## Domain-Specific Context

### TDA Algorithms
- **Persistent Homology**: Core mathematical foundation for topological feature extraction
- **Filtration Types**: Vietoris-Rips, Alpha Complex, Čech, DTM
- **Vectorization**: Persistence landscapes, persistence images, Betti curves
- **Mapper Algorithm**: Topological network construction with customizable parameters

### Finance Module
- **Market Regime Detection**: Identify structural changes in financial time series
- **Time-Series Embedding**: Convert 1D data to point clouds for topological analysis
- **Early Warning Signals**: Configurable alerts for regime transitions
- **Backtesting Framework**: Historical validation of detection algorithms

### Cybersecurity Module
- **Network Traffic Analysis**: Real-time packet stream processing
- **Anomaly Detection**: Baseline profiling with deviation detection
- **Attack Classification**: Supervised learning for threat categorization
- **Real-time Processing**: <100ms latency for threat detection

## Testing & Quality Standards

### Code Quality
- **Test Coverage**: Minimum 80% for critical paths
- **Static Analysis**: Use clang-tidy, mypy, ESLint
- **Performance Testing**: Benchmark against 1M+ point datasets
- **Security Testing**: Penetration testing, vulnerability scanning

### Documentation
- **API Documentation**: OpenAPI/Swagger with examples
- **Algorithm Documentation**: Mathematical foundations and implementation details
- **User Guides**: End-to-end workflow documentation
- **Developer Setup**: Comprehensive environment setup instructions

## Common Issues & Anti-Patterns

### ❌ Avoid
- **Memory Leaks**: Raw pointers, manual memory management
- **Blocking Operations**: Synchronous I/O in async contexts
- **Hard-coded Values**: Magic numbers, configuration in code
- **Global State**: Singletons, mutable global variables
- **Tight Coupling**: Direct dependencies between modules

### ✅ Prefer
- **RAII Patterns**: Automatic resource management
- **Async Operations**: Non-blocking I/O with proper error handling
- **Configuration Files**: Environment-based configuration
- **Dependency Injection**: Loose coupling through interfaces
- **Event-Driven Architecture**: Decoupled communication patterns

## Review Priorities

### High Priority
- **Security**: Authentication, authorization, data protection
- **Performance**: Algorithm efficiency, memory usage, scalability
- **Data Integrity**: Transaction management, consistency patterns
- **Error Handling**: Proper exception handling, user feedback

### Medium Priority
- **Code Quality**: Readability, maintainability, test coverage
- **Documentation**: API docs, inline comments, user guides
- **Testing**: Unit tests, integration tests, performance tests

### Low Priority
- **Code Style**: Formatting, naming conventions (unless egregious)
- **Minor Optimizations**: Premature optimization without profiling

## Project-Specific File Patterns

### C++ Files
- **Headers**: `.hpp` extension, include guards, forward declarations
- **Implementation**: `.cpp` extension, implementation details
- **Templates**: Header-only when possible, explicit instantiation when needed

### Python Files
- **Modules**: `__init__.py` files, proper package structure
- **Services**: Single responsibility, dependency injection
- **API**: FastAPI decorators, proper response models

### Frontend Files
- **Components**: Functional components with hooks, proper prop types
- **Services**: API client patterns, error handling
- **Visualizations**: D3.js/Three.js integration, responsive design

## Integration Points

### C++ ↔ Python
- **pybind11 Bindings**: Proper memory management, exception translation
- **Data Types**: NumPy arrays, Python objects, C++ containers
- **Performance**: Minimize Python overhead for critical paths

### Backend ↔ Frontend
- **API Contracts**: Consistent request/response formats
- **Real-time Updates**: WebSocket connections, server-sent events
- **Error Handling**: Proper HTTP status codes, user-friendly messages

### Database ↔ Services
- **Connection Pooling**: Efficient database connection management
- **Transaction Boundaries**: Proper ACID compliance
- **Data Migration**: Versioned schema changes, rollback strategies
