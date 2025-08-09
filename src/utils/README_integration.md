# TDA Vector Stack Integration Utilities

This directory contains a comprehensive set of integration utilities that bridge the existing TDA infrastructure with the new vector stack implementation. These utilities ensure seamless interoperability, robust error handling, and consistent data flow across the entire pipeline.

## Overview

The integration system consists of six core components that work together to enable vector stack processing on existing diagram dumps:

1. **Manifest Splitter** (`manifest_splitter.py`) - Temporal train/test splitting
2. **Path Resolver** (`path_resolver.py`) - Cross-context path resolution
3. **Configuration Mapper** (`config_mapper.py`) - Parameter alignment and validation
4. **NPZ Validator** (`npz_validator.py`) - Data format validation and verification
5. **Integration Tester** (`integration_tester.py`) - End-to-end pipeline testing
6. **Error Handler** (`error_handler.py`) - Graceful degradation and recovery
7. **Integration Bridge** (`integration_bridge.py`) - Unified orchestration interface

## Quick Start

### Basic Usage

```python
from src.utils.integration_bridge import run_integration_bridge

# Run complete integration pipeline
results = run_integration_bridge(
    manifest_path="validation/diagrams/diagram_manifest.json",
    output_dir="output/vector_stack_results",
    dataset_config_path="configs/dataset_cic.yaml"  # optional
)

print(f"Success: {results.success}")
print(f"Train samples: {results.manifest_split_stats['counts']['train']}")
print(f"Test samples: {results.manifest_split_stats['counts']['test']}")
```

### Command Line Usage

```bash
# Run integration bridge from command line
python src/utils/integration_bridge.py \
    validation/diagrams/diagram_manifest.json \
    output/integration_results \
    configs/dataset_cic.yaml

# Test individual components
python src/utils/manifest_splitter.py manifest.json output_dir 0.7
python src/utils/npz_validator.py manifest.json validation_report.json
python src/utils/integration_tester.py manifest.json dataset_config.yaml test_report.json
```

## Component Details

### 1. Manifest Splitter

Converts unified diagram manifests into temporally-separated train/test splits while preserving metadata integrity.

**Key Features:**
- Temporal boundary enforcement (zero leakage)
- Configurable split ratios with minimum sample requirements
- Label distribution preservation
- Metadata consistency across splits

**Usage:**
```python
from src.utils.manifest_splitter import ManifestSplitter, SplitConfig

config = SplitConfig(train_ratio=0.7, min_test_samples=50)
splitter = ManifestSplitter(config)
stats = splitter.split_manifest(manifest_path, output_dir)
```

### 2. Path Resolver

Handles path resolution across different execution contexts, ensuring compatibility between relative paths in manifests and absolute paths required by processing.

**Key Features:**
- Context-aware path resolution (script directory, working directory)
- Multiple fallback strategies
- Batch processing for manifest entries
- Cross-platform compatibility

**Usage:**
```python
from src.utils.path_resolver import PathResolver, resolve_diagram_path

# Single path resolution
resolved_path = resolve_diagram_path("window_000001.npz", manifest_dir)

# Batch processing
resolver = PathResolver()
resolved_entries = resolver.resolve_manifest_entries(entries, manifest_path)
```

### 3. Configuration Mapper

Maps between different configuration formats used across the TDA system, ensuring parameter consistency and providing validation.

**Key Features:**
- Bi-directional mapping between config formats (YAML ↔ VectorStackConfig)
- Parameter validation with type checking
- Configuration compatibility checking
- Export/import capabilities

**Usage:**
```python
from src.utils.config_mapper import ConfigMapper

mapper = ConfigMapper()
pipeline_config = mapper.create_pipeline_config(
    dataset_config_path, output_dir, manifest_path=manifest_path
)
issues = mapper.validate_config_compatibility(pipeline_config)
```

### 4. NPZ Validator

Validates NPZ files containing persistence diagrams against expected formats, ensuring compatibility between existing dumps and vector stack processing.

**Key Features:**
- Schema validation for diagram NPZ files
- Key naming consistency checks (dgm_H0, dgm_H1, etc.)
- Data type and shape validation
- Finite value verification
- Batch validation with detailed reporting

**Usage:**
```python
from src.utils.npz_validator import NPZValidator, validate_diagram_manifest

# Validate single file
validator = NPZValidator()
result = validator.validate_file(npz_path)

# Validate entire manifest
report = validate_diagram_manifest(manifest_path)
```

### 5. Integration Tester

Provides comprehensive end-to-end testing of the integration pipeline to ensure all components work together correctly.

**Key Features:**
- NPZ format validation testing
- Path resolution testing across contexts
- Configuration mapping validation
- Vector stack feature extraction testing
- Complete pipeline integration testing

**Usage:**
```python
from src.utils.integration_tester import run_integration_test

report = run_integration_test(
    manifest_path, dataset_config_path, 
    output_report_path="integration_test_report.json"
)
print(f"Success rate: {report['summary']['success_rate']:.1%}")
```

### 6. Error Handler

Provides robust error handling with fallback strategies to ensure the pipeline can continue operating even when individual components fail.

**Key Features:**
- Graceful degradation strategies
- Comprehensive error logging and reporting
- Recovery mechanisms for common failure modes
- Context-aware error messages with troubleshooting guidance

**Usage:**
```python
from src.utils.error_handler import ErrorHandler, safe_operation

# Automatic error handling with decorator
@safe_operation("my_component", fallback_value=[])
def process_data(data):
    return expensive_operation(data)

# Manual error handling
handler = ErrorHandler()
result = handler.safe_file_load(file_path, "loader", json.load, fallback_value={})
```

### 7. Integration Bridge (Main Interface)

Orchestrates all integration components to provide a unified interface for vector stack processing on existing TDA infrastructure.

**Key Features:**
- Complete pipeline orchestration
- Automatic fallback and error recovery
- Comprehensive validation and reporting
- Configurable processing pipeline
- Detailed execution logging

## Configuration Options

### BridgeConfig

Main configuration class for the integration bridge:

```python
@dataclass
class BridgeConfig:
    # Validation settings
    validate_npz_files: bool = True
    validate_paths: bool = True 
    validate_configs: bool = True
    run_integration_tests: bool = False  # Expensive
    
    # Splitting settings
    train_ratio: float = 0.7
    min_test_samples: int = 50
    min_train_samples: int = 100
    temporal_gap_seconds: float = 0.0
    
    # Processing settings
    max_samples_for_testing: int = 100
    enable_feature_extraction: bool = True
    strict_error_handling: bool = True
    
    # Output settings
    export_detailed_logs: bool = True
    create_backup_manifests: bool = True
```

## Error Handling Strategies

The integration system implements multiple levels of error handling:

1. **Component-Level**: Each utility handles its own errors with appropriate fallbacks
2. **Pipeline-Level**: The integration bridge coordinates error recovery across components
3. **Data-Level**: Graceful handling of missing/corrupt files with sensible defaults
4. **Configuration-Level**: Validation and correction of parameter mismatches

## Common Integration Patterns

### Pattern 1: Existing Unified Manifest → Vector Stack Features

```python
# Start with unified manifest, get train/test features
results = run_integration_bridge(
    "validation/diagrams/unified_manifest.json",
    "output/vector_features",
    enable_feature_extraction=True
)

# Features are saved as NPZ files
train_data = np.load(results.output_paths['train_features'])
X_train, y_train = train_data['X'], train_data['y']
```

### Pattern 2: Configuration Migration

```python
# Convert dataset config to vector stack config
mapper = ConfigMapper()
pipeline_config = mapper.create_pipeline_config(
    "configs/dataset_cic.yaml", 
    "output/migrated_pipeline"
)

# Validate compatibility
issues = mapper.validate_config_compatibility(pipeline_config)
if issues['errors']:
    print("Configuration errors:", issues['errors'])
```

### Pattern 3: Robust File Processing

```python
# Process all diagrams with error handling
handler = ErrorHandler()
successful_results, failed_entries = handler.safe_manifest_processing(
    manifest_entries, 
    process_diagram_entry,
    "diagram_processor"
)

# Get error summary
summary = handler.get_error_summary()
print(f"Processed {len(successful_results)} successfully, {len(failed_entries)} failed")
```

## Output Structure

The integration bridge creates a structured output directory:

```
output_directory/
├── manifests/
│   ├── train_manifest.json      # Temporal train split
│   ├── test_manifest.json       # Temporal test split
│   └── split_stats.json         # Split statistics
├── features/                    # Vector stack features (if enabled)
│   ├── train_features.npz       # Training feature vectors
│   ├── test_features.npz        # Test feature vectors
│   └── kernel_dicts.npz         # Kernel dictionaries
├── logs/
│   ├── bridge_results.json      # Complete execution results
│   ├── error_log.json           # Detailed error log
│   └── bridge_config.json       # Configuration used
└── pipeline_config.json         # Full pipeline configuration
```

## Best Practices

1. **Always validate before processing**: Use NPZ validator and path resolver before feature extraction
2. **Handle errors gracefully**: Use error handling utilities rather than raw try/catch
3. **Preserve temporal integrity**: Always check for temporal leakage in splits
4. **Use configuration validation**: Validate parameter compatibility before running pipeline
5. **Enable detailed logging**: Keep comprehensive logs for troubleshooting
6. **Test integration thoroughly**: Run integration tests when making changes

## Troubleshooting

### Common Issues

1. **Path Resolution Failures**
   - Check that NPZ files exist in expected locations
   - Verify manifest file field names ('file' vs 'diagram_path')
   - Use path resolver validation to identify missing files

2. **Configuration Mismatches**
   - Validate config compatibility before pipeline execution  
   - Check homology dimension consistency
   - Verify random seed alignment

3. **NPZ Format Issues**
   - Run NPZ validator to identify format problems
   - Check for expected keys (dgm_H0, dgm_H1, etc.)
   - Verify data types and shapes

4. **Temporal Leakage**
   - Ensure manifest entries are properly sorted by timestamp
   - Check split configuration parameters
   - Verify temporal gap requirements

### Debugging Tools

```python
# Enable detailed error logging
config = BridgeConfig(
    export_detailed_logs=True,
    run_integration_tests=True,  # Comprehensive testing
    strict_error_handling=False  # Don't fail on minor errors
)

# Run with debugging enabled
results = run_integration_bridge(manifest_path, output_dir, config=config)

# Check results
if not results.success:
    print("Errors:", results.error_summary)
    print("Warnings:", results.warnings)
```

## Integration with Existing Scripts

The integration utilities are designed to work with existing TDA scripts:

- **`extract_vector_stack.py`**: Now automatically uses integration bridge for manifest preparation
- **`baseline_gbt_training.py`**: Can use path resolver for robust file loading
- **Custom analysis scripts**: Can use error handler for graceful degradation

## Performance Considerations

- **NPZ validation**: Can be expensive for large manifests; consider sampling
- **Integration testing**: Full tests are comprehensive but slow; disable for production
- **Feature extraction**: Limited by `max_samples_for_testing` for faster validation
- **Path resolution**: Caches results to avoid repeated file system checks

## Extensions

The integration system is designed to be extensible:

1. **Custom Error Handlers**: Implement domain-specific recovery strategies
2. **Additional Validators**: Add validation for new data formats
3. **Enhanced Path Resolution**: Add support for remote file systems
4. **Configuration Mappers**: Support for additional config formats
5. **Integration Tests**: Add tests for new pipeline components

## Support

For issues with the integration utilities:

1. Check the error logs in `output_dir/logs/`
2. Run integration tests to identify specific component failures
3. Use individual utility validation functions to isolate issues
4. Review configuration compatibility reports for parameter mismatches

The integration system is designed to provide clear error messages and recovery guidance to help resolve issues quickly.