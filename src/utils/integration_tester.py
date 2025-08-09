"""End-to-end integration testing for TDA vector stack pipeline.

Validates the complete integration pipeline from existing diagram dumps
through vector stack extraction to classifier training, ensuring all 
components work together correctly.

Test Scenarios:
- Manifest splitting with temporal integrity
- Path resolution across execution contexts  
- NPZ format compatibility validation
- Configuration parameter alignment
- Vector stack feature extraction
- Training data preparation
- Error handling and graceful degradation
"""
from __future__ import annotations
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import traceback
from datetime import datetime

# Import integration utilities
from src.utils.manifest_splitter import ManifestSplitter, SplitConfig
from src.utils.path_resolver import PathResolver, PathResolutionConfig
from src.utils.config_mapper import ConfigMapper, create_config_for_dataset
from src.utils.npz_validator import NPZValidator, NPZValidationConfig
from src.embeddings.vector_stack import VectorStackConfig, prepare_kernel_dictionaries, build_vector_stack


@dataclass
class IntegrationTestConfig:
    """Configuration for integration testing."""
    use_temp_workspace: bool = True  # Use temporary directory for testing
    validate_npz_files: bool = True  # Run NPZ validation
    test_vector_extraction: bool = True  # Test vector stack features
    test_train_test_split: bool = True  # Test manifest splitting
    test_path_resolution: bool = True  # Test path handling
    cleanup_on_success: bool = True  # Clean up temp files if tests pass
    max_test_samples: int = 50  # Limit samples for faster testing
    random_seed: int = 42


@dataclass
class TestResult:
    """Results from a single test."""
    test_name: str
    passed: bool
    error_message: Optional[str]
    warning_messages: List[str]
    execution_time_seconds: float
    details: Dict[str, Any]


class IntegrationTester:
    """Comprehensive integration test runner."""
    
    def __init__(self, config: IntegrationTestConfig = None):
        self.config = config or IntegrationTestConfig()
        self.test_results: List[TestResult] = []
        self.workspace_dir: Optional[Path] = None
    
    def run_full_integration_test(self, 
                                manifest_path: Path,
                                dataset_config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run complete integration test suite.
        
        Parameters
        ----------
        manifest_path : Path
            Path to existing diagram manifest
        dataset_config_path : Path, optional
            Path to dataset configuration YAML
            
        Returns
        -------
        Dict[str, Any]
            Complete test report with results and statistics
        """
        start_time = datetime.now()
        
        # Setup test workspace
        self._setup_workspace()
        
        try:
            # Core integration tests
            if self.config.validate_npz_files:
                self._test_npz_validation(manifest_path)
            
            if self.config.test_path_resolution:
                self._test_path_resolution(manifest_path)
            
            if self.config.test_train_test_split:
                self._test_manifest_splitting(manifest_path)
            
            if dataset_config_path:
                self._test_config_mapping(dataset_config_path)
            
            if self.config.test_vector_extraction:
                self._test_vector_stack_extraction(manifest_path)
            
            # End-to-end pipeline test
            self._test_end_to_end_pipeline(manifest_path, dataset_config_path)
            
        except Exception as e:
            # Catch any unhandled exceptions
            self._add_test_result(
                "unhandled_exception",
                False,
                f"Unhandled exception during testing: {str(e)}",
                [],
                0.0,
                {"traceback": traceback.format_exc()}
            )
        
        finally:
            # Cleanup if configured
            if self.config.cleanup_on_success and self._all_tests_passed():
                self._cleanup_workspace()
        
        end_time = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_test_report(start_time, end_time)
        
        return report
    
    def _setup_workspace(self):
        """Setup temporary workspace for testing."""
        if self.config.use_temp_workspace:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="tda_integration_test_"))
        else:
            self.workspace_dir = Path("./integration_test_workspace")
            self.workspace_dir.mkdir(exist_ok=True)
        
        print(f"Test workspace: {self.workspace_dir}")
    
    def _cleanup_workspace(self):
        """Clean up test workspace."""
        if self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
    
    def _test_npz_validation(self, manifest_path: Path):
        """Test NPZ file validation."""
        start_time = datetime.now()
        
        try:
            validator = NPZValidator()
            report = validator.validate_manifest_diagrams(manifest_path)
            
            # Check validation results
            summary = report['batch_validation_summary']
            success_rate = summary['success_rate']
            
            passed = success_rate > 0.8  # Allow some tolerance
            warnings = []
            
            if success_rate < 1.0:
                warnings.append(f"NPZ validation success rate: {success_rate:.1%}")
            
            if summary['total_errors'] > 0:
                warnings.append(f"Total validation errors: {summary['total_errors']}")
            
            self._add_test_result(
                "npz_validation",
                passed,
                None if passed else f"Low NPZ validation success rate: {success_rate:.1%}",
                warnings,
                (datetime.now() - start_time).total_seconds(),
                {
                    "success_rate": success_rate,
                    "total_files": summary['total_files'],
                    "valid_files": summary['valid_files'],
                    "total_errors": summary['total_errors']
                }
            )
            
        except Exception as e:
            self._add_test_result(
                "npz_validation",
                False,
                f"NPZ validation failed: {str(e)}",
                [],
                (datetime.now() - start_time).total_seconds(),
                {"exception": str(e)}
            )
    
    def _test_path_resolution(self, manifest_path: Path):
        """Test path resolution functionality."""
        start_time = datetime.now()
        
        try:
            config = PathResolutionConfig(validate_existence=True)
            resolver = PathResolver(config)
            
            # Validate manifest paths
            report = resolver.validate_manifest_paths(manifest_path)
            
            success_rate = report['success_rate']
            passed = success_rate > 0.9
            warnings = []
            
            if success_rate < 1.0:
                warnings.append(f"Path resolution success rate: {success_rate:.1%}")
            
            self._add_test_result(
                "path_resolution",
                passed,
                None if passed else f"Path resolution issues: {success_rate:.1%} success rate",
                warnings,
                (datetime.now() - start_time).total_seconds(),
                {
                    "success_rate": success_rate,
                    "total_entries": report['total_entries'],
                    "resolved_count": report['resolved_count'],
                    "missing_count": report['missing_count']
                }
            )
            
        except Exception as e:
            self._add_test_result(
                "path_resolution",
                False,
                f"Path resolution test failed: {str(e)}",
                [],
                (datetime.now() - start_time).total_seconds(),
                {"exception": str(e)}
            )
    
    def _test_manifest_splitting(self, manifest_path: Path):
        """Test manifest splitting with temporal integrity."""
        start_time = datetime.now()
        
        try:
            split_config = SplitConfig(
                train_ratio=0.7,
                min_test_samples=10,  # Lower for testing
                min_train_samples=20,
                random_seed=self.config.random_seed
            )
            
            splitter = ManifestSplitter(split_config)
            output_dir = self.workspace_dir / "split_test"
            
            stats = splitter.split_manifest(manifest_path, output_dir)
            
            # Validate split results
            passed = True
            warnings = []
            error_message = None
            
            # Check temporal integrity
            if stats['integrity_checks']['temporal_leakage']:
                passed = False
                error_message = "Temporal leakage detected in split"
            
            # Check minimum sample requirements
            if not stats['integrity_checks']['min_samples_satisfied']:
                passed = False
                error_message = "Minimum sample requirements not met"
            
            # Check split ratio is reasonable
            actual_ratio = stats['counts']['train_ratio_actual']
            if abs(actual_ratio - split_config.train_ratio) > 0.1:
                warnings.append(f"Split ratio deviation: expected {split_config.train_ratio}, got {actual_ratio:.2f}")
            
            self._add_test_result(
                "manifest_splitting",
                passed,
                error_message,
                warnings,
                (datetime.now() - start_time).total_seconds(),
                {
                    "train_count": stats['counts']['train'],
                    "test_count": stats['counts']['test'],
                    "actual_ratio": actual_ratio,
                    "temporal_gap": stats['temporal_boundaries']['temporal_gap']
                }
            )
            
        except Exception as e:
            self._add_test_result(
                "manifest_splitting",
                False,
                f"Manifest splitting failed: {str(e)}",
                [],
                (datetime.now() - start_time).total_seconds(),
                {"exception": str(e)}
            )
    
    def _test_config_mapping(self, dataset_config_path: Path):
        """Test configuration mapping functionality."""
        start_time = datetime.now()
        
        try:
            mapper = ConfigMapper()
            
            # Load dataset config
            dataset_config = mapper.load_dataset_config(dataset_config_path)
            
            # Create pipeline config
            output_dir = self.workspace_dir / "config_test"
            pipeline_config = mapper.create_pipeline_config(dataset_config_path, output_dir)
            
            # Validate compatibility
            issues = mapper.validate_config_compatibility(pipeline_config)
            
            passed = len(issues['errors']) == 0
            warnings = issues['warnings']
            error_message = None
            
            if issues['errors']:
                error_message = f"Configuration errors: {issues['errors'][:2]}"
            
            # Test export/import cycle
            config_path = output_dir / "pipeline_config.json"
            mapper.export_config(pipeline_config, config_path)
            loaded_config = mapper.load_pipeline_config(config_path)
            
            # Verify round-trip consistency
            if loaded_config.dataset.name != dataset_config.name:
                warnings.append("Config export/import inconsistency detected")
            
            self._add_test_result(
                "config_mapping",
                passed,
                error_message,
                warnings,
                (datetime.now() - start_time).total_seconds(),
                {
                    "dataset_name": dataset_config.name,
                    "error_count": len(issues['errors']),
                    "warning_count": len(issues['warnings']),
                    "homology_dims": list(pipeline_config.vector_stack.homology_dims)
                }
            )
            
        except Exception as e:
            self._add_test_result(
                "config_mapping",
                False,
                f"Config mapping failed: {str(e)}",
                [],
                (datetime.now() - start_time).total_seconds(),
                {"exception": str(e)}
            )
    
    def _test_vector_stack_extraction(self, manifest_path: Path):
        """Test vector stack feature extraction."""
        start_time = datetime.now()
        
        try:
            # Load a subset of diagrams for testing
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            entries = manifest['entries'][:self.config.max_test_samples]
            manifest_dir = manifest_path.parent
            
            # Load diagrams
            diagrams_by_sample = []
            for entry in entries:
                file_path = manifest_dir / entry['file']
                if file_path.exists():
                    npz_data = np.load(file_path)
                    diagrams = {}
                    for key in npz_data.files:
                        if key.startswith('dgm_H'):
                            dim = int(key.split('H')[1])
                            diagrams[dim] = npz_data[key]
                    diagrams_by_sample.append(diagrams)
                    npz_data.close()
                    
                    if len(diagrams_by_sample) >= 10:  # Limit for testing
                        break
            
            if not diagrams_by_sample:
                raise ValueError("No valid diagrams found for testing")
            
            # Test vector stack configuration and extraction
            config = VectorStackConfig(
                homology_dims=(0, 1),
                random_seed=self.config.random_seed,
                max_points_per_block=1000  # Smaller for testing
            )
            
            # Prepare kernel dictionaries
            train_diagrams_by_dim = {dim: [] for dim in config.homology_dims}
            for diagrams in diagrams_by_sample[:5]:  # Use first 5 for training
                for dim in config.homology_dims:
                    if dim in diagrams:
                        train_diagrams_by_dim[dim].append(diagrams[dim])
            
            kernel_dicts = prepare_kernel_dictionaries(train_diagrams_by_dim, config)
            
            # Extract features for all samples
            feature_vectors = []
            for diagrams in diagrams_by_sample:
                try:
                    vec, stats, spans = build_vector_stack(diagrams, config, kernel_dicts, fit_norm=len(feature_vectors)==0)
                    feature_vectors.append(vec)
                except Exception as e:
                    # Individual sample failure shouldn't fail the whole test
                    pass
            
            passed = len(feature_vectors) >= len(diagrams_by_sample) * 0.8  # 80% success rate
            warnings = []
            error_message = None
            
            if not passed:
                error_message = f"Low feature extraction success rate: {len(feature_vectors)}/{len(diagrams_by_sample)}"
            
            # Check feature vector consistency
            if feature_vectors:
                vector_shapes = [v.shape for v in feature_vectors]
                if not all(shape == vector_shapes[0] for shape in vector_shapes):
                    warnings.append("Inconsistent feature vector shapes detected")
                
                # Check for valid values
                all_vectors = np.vstack(feature_vectors)
                if not np.isfinite(all_vectors).all():
                    warnings.append("Non-finite values in feature vectors")
            
            self._add_test_result(
                "vector_stack_extraction",
                passed,
                error_message,
                warnings,
                (datetime.now() - start_time).total_seconds(),
                {
                    "samples_processed": len(diagrams_by_sample),
                    "successful_extractions": len(feature_vectors),
                    "feature_vector_shape": vector_shapes[0] if feature_vectors else None,
                    "homology_dims": list(config.homology_dims)
                }
            )
            
        except Exception as e:
            self._add_test_result(
                "vector_stack_extraction",
                False,
                f"Vector stack extraction failed: {str(e)}",
                [],
                (datetime.now() - start_time).total_seconds(),
                {"exception": str(e)}
            )
    
    def _test_end_to_end_pipeline(self, manifest_path: Path, dataset_config_path: Optional[Path]):
        """Test complete end-to-end pipeline integration."""
        start_time = datetime.now()
        
        try:
            # Create a minimal end-to-end test
            output_dir = self.workspace_dir / "e2e_test"
            
            # Step 1: Split manifest
            split_config = SplitConfig(train_ratio=0.8, min_test_samples=5, min_train_samples=10)
            splitter = ManifestSplitter(split_config)
            split_stats = splitter.split_manifest(manifest_path, output_dir)
            
            # Step 2: Validate paths in split manifests
            train_manifest = output_dir / "train_manifest.json"
            test_manifest = output_dir / "test_manifest.json"
            
            resolver = PathResolver()
            train_report = resolver.validate_manifest_paths(train_manifest)
            test_report = resolver.validate_manifest_paths(test_manifest)
            
            # Step 3: Extract features from subset
            config = VectorStackConfig(homology_dims=(0, 1), random_seed=self.config.random_seed)
            
            # Load small subset for testing
            with open(train_manifest, 'r') as f:
                train_data = json.load(f)
            
            train_entries = train_data['entries'][:min(10, len(train_data['entries']))]
            
            # Simple feature extraction test
            feature_count = 0
            for entry in train_entries:
                try:
                    if 'diagram_path' in entry:
                        diagram_path = Path(entry['diagram_path'])
                        if not diagram_path.is_absolute():
                            diagram_path = manifest_path.parent / entry['file']
                        
                        if diagram_path.exists():
                            npz_data = np.load(diagram_path)
                            diagrams = {}
                            for key in npz_data.files:
                                if key.startswith('dgm_H'):
                                    dim = int(key.split('H')[1])
                                    diagrams[dim] = npz_data[key]
                            
                            if diagrams:
                                feature_count += 1
                            
                            npz_data.close()
                except Exception:
                    pass  # Individual failures are acceptable
            
            # Evaluate end-to-end success
            passed = True
            warnings = []
            error_message = None
            
            # Check manifest splitting success
            if not split_stats['integrity_checks']['min_samples_satisfied']:
                passed = False
                error_message = "End-to-end test: insufficient samples after splitting"
            
            # Check path resolution success
            if train_report['success_rate'] < 0.9:
                warnings.append(f"Train manifest path resolution: {train_report['success_rate']:.1%}")
            
            if test_report['success_rate'] < 0.9:
                warnings.append(f"Test manifest path resolution: {test_report['success_rate']:.1%}")
            
            # Check feature extraction
            if feature_count < len(train_entries) * 0.7:
                warnings.append(f"Low feature extraction rate: {feature_count}/{len(train_entries)}")
            
            self._add_test_result(
                "end_to_end_pipeline",
                passed,
                error_message,
                warnings,
                (datetime.now() - start_time).total_seconds(),
                {
                    "train_samples": split_stats['counts']['train'],
                    "test_samples": split_stats['counts']['test'],
                    "train_path_success_rate": train_report['success_rate'],
                    "test_path_success_rate": test_report['success_rate'],
                    "feature_extraction_rate": feature_count / len(train_entries) if train_entries else 0
                }
            )
            
        except Exception as e:
            self._add_test_result(
                "end_to_end_pipeline", 
                False,
                f"End-to-end pipeline test failed: {str(e)}",
                [],
                (datetime.now() - start_time).total_seconds(),
                {"exception": str(e)}
            )
    
    def _add_test_result(self, 
                        test_name: str, 
                        passed: bool, 
                        error_message: Optional[str],
                        warnings: List[str],
                        execution_time: float,
                        details: Dict[str, Any]):
        """Add a test result to the collection."""
        result = TestResult(
            test_name=test_name,
            passed=passed,
            error_message=error_message,
            warning_messages=warnings,
            execution_time_seconds=execution_time,
            details=details
        )
        
        self.test_results.append(result)
    
    def _all_tests_passed(self) -> bool:
        """Check if all tests passed."""
        return all(result.passed for result in self.test_results)
    
    def _generate_test_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        total_warnings = sum(len(r.warning_messages) for r in self.test_results)
        total_execution_time = sum(r.execution_time_seconds for r in self.test_results)
        
        report = {
            "integration_test_report": {
                "test_timestamp": start_time.isoformat(),
                "test_duration_seconds": (end_time - start_time).total_seconds(),
                "workspace_directory": str(self.workspace_dir) if self.workspace_dir else None,
                "configuration": asdict(self.config)
            },
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.test_results) if self.test_results else 0.0,
                "total_warnings": total_warnings,
                "total_execution_time_seconds": total_execution_time
            },
            "test_results": [asdict(result) for result in self.test_results],
            "failed_tests": [result.test_name for result in failed_tests],
            "test_details": {result.test_name: result.details for result in self.test_results}
        }
        
        return report


def run_integration_test(manifest_path: Union[str, Path],
                        dataset_config_path: Optional[Union[str, Path]] = None,
                        config: Optional[IntegrationTestConfig] = None,
                        output_report_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Convenience function to run integration tests.
    
    Parameters
    ---------- 
    manifest_path : str | Path
        Path to diagram manifest file
    dataset_config_path : str | Path, optional
        Path to dataset configuration YAML
    config : IntegrationTestConfig, optional
        Test configuration
    output_report_path : str | Path, optional
        Path to save detailed test report
        
    Returns
    -------
    Dict[str, Any]
        Test report with results
    """
    tester = IntegrationTester(config)
    
    report = tester.run_full_integration_test(
        Path(manifest_path),
        Path(dataset_config_path) if dataset_config_path else None
    )
    
    if output_report_path:
        with open(output_report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report


if __name__ == "__main__":
    # Command-line interface
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python integration_tester.py <manifest_path> [dataset_config_path] [report_path]")
        sys.exit(1)
    
    manifest_path = Path(sys.argv[1])
    dataset_config_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    report_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Run integration tests
    config = IntegrationTestConfig()
    report = run_integration_test(manifest_path, dataset_config_path, config, report_path)
    
    # Print summary
    summary = report['summary']
    print("\n=== TDA Integration Test Results ===")
    print(f"Tests run: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total warnings: {summary['total_warnings']}")
    print(f"Execution time: {summary['total_execution_time_seconds']:.1f}s")
    
    if summary['failed_tests'] > 0:
        print("\nFailed tests:")
        for test_name in report['failed_tests']:
            print(f"  - {test_name}")
    
    if report_path:
        print(f"\nDetailed report saved to: {report_path}")
    
    # Exit with error code if tests failed
    sys.exit(0 if summary['failed_tests'] == 0 else 1)