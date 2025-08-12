"""Complete integration bridge for TDA vector stack pipeline.

Provides a unified interface that combines all integration utilities to create
a seamless bridge between existing TDA infrastructure and vector stack processing.

This bridge handles:
- Automatic manifest splitting and validation
- Path resolution across execution contexts
- Configuration alignment and validation
- NPZ format validation and error recovery
- End-to-end pipeline orchestration
- Comprehensive error handling and reporting

Usage:
    bridge = IntegrationBridge()
    results = bridge.run_complete_pipeline(manifest_path, dataset_config_path, output_dir)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Import all integration utilities
from src.utils.manifest_splitter import ManifestSplitter, SplitConfig
from src.utils.path_resolver import PathResolver, PathResolutionConfig
from src.utils.config_mapper import ConfigMapper, PipelineConfig
from src.utils.npz_validator import NPZValidator, NPZValidationConfig
from src.utils.integration_tester import IntegrationTester, IntegrationTestConfig
from src.utils.error_handler import ErrorHandler, ErrorHandlingConfig, ErrorSeverity
from src.embeddings.vector_stack import VectorStackConfig, prepare_kernel_dictionaries, build_vector_stack


@dataclass
class BridgeConfig:
    """Configuration for the integration bridge."""
    # Validation settings
    validate_npz_files: bool = True
    validate_paths: bool = True 
    validate_configs: bool = True
    run_integration_tests: bool = False  # Expensive, only for debugging
    
    # Splitting settings
    train_ratio: float = 0.7
    min_test_samples: int = 50
    min_train_samples: int = 100
    temporal_gap_seconds: float = 0.0
    
    # Processing settings
    max_samples_for_testing: int = 100  # Limit samples during validation
    enable_feature_extraction: bool = True
    strict_error_handling: bool = True
    
    # Output settings
    export_detailed_logs: bool = True
    create_backup_manifests: bool = True
    
    # Performance settings
    random_seed: int = 1337


@dataclass 
class BridgeResults:
    """Results from complete bridge pipeline execution."""
    success: bool
    execution_time_seconds: float
    manifest_split_stats: Dict[str, Any]
    validation_reports: Dict[str, Any]
    config_validation: Dict[str, Any]
    feature_extraction_stats: Dict[str, Any]
    error_summary: Dict[str, Any]
    output_paths: Dict[str, str]
    warnings: List[str]


class IntegrationBridge:
    """Main integration bridge orchestrating all components."""
    
    def __init__(self, config: BridgeConfig = None):
        self.config = config or BridgeConfig()
        self.error_handler = ErrorHandler(ErrorHandlingConfig(
            raise_on_critical=self.config.strict_error_handling,
            enable_recovery_logging=True
        ))
        
        # Initialize all utility components
        self.manifest_splitter = ManifestSplitter(SplitConfig(
            train_ratio=self.config.train_ratio,
            min_test_samples=self.config.min_test_samples,
            min_train_samples=self.config.min_train_samples,
            temporal_gap_seconds=self.config.temporal_gap_seconds,
            random_seed=self.config.random_seed
        ))
        
        self.path_resolver = PathResolver(PathResolutionConfig(
            validate_existence=self.config.validate_paths,
            allow_relative_fallback=True
        ))
        
        self.config_mapper = ConfigMapper()
        
        self.npz_validator = NPZValidator(NPZValidationConfig(
            require_finite_values=True,
            allow_empty_diagrams=True,
            check_birth_death_order=True
        ))
        
        if self.config.run_integration_tests:
            self.integration_tester = IntegrationTester(IntegrationTestConfig(
                use_temp_workspace=True,
                max_test_samples=self.config.max_samples_for_testing
            ))
    
    def run_complete_pipeline(self, 
                             manifest_path: Union[str, Path],
                             dataset_config_path: Optional[Union[str, Path]],
                             output_dir: Union[str, Path]) -> BridgeResults:
        """Execute complete integration pipeline.
        
        Parameters
        ----------
        manifest_path : str | Path
            Path to existing unified diagram manifest
        dataset_config_path : str | Path, optional
            Path to dataset configuration YAML
        output_dir : str | Path
            Directory for pipeline outputs
            
        Returns
        -------
        BridgeResults
            Complete pipeline execution results
        """
        start_time = datetime.now()
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)
        dataset_config_path = Path(dataset_config_path) if dataset_config_path else None
        
        # Create output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        warnings = []
        
        try:
            # Phase 1: Initial validation
            self.error_handler.logger.info("Starting TDA vector stack integration pipeline")
            validation_reports = self._run_initial_validation(manifest_path)
            
            if validation_reports['npz_validation']['batch_validation_summary']['success_rate'] < 0.8:
                warnings.append("Low NPZ validation success rate - some diagrams may be problematic")
            
            # Phase 2: Configuration setup
            if dataset_config_path:
                pipeline_config = self._setup_pipeline_configuration(
                    dataset_config_path, output_dir, manifest_path
                )
                config_validation = self.config_mapper.validate_config_compatibility(pipeline_config)
                
                if config_validation['errors']:
                    raise ValueError(f"Configuration validation errors: {config_validation['errors']}")
                
                warnings.extend(config_validation['warnings'])
            else:
                pipeline_config = None
                config_validation = {"warnings": [], "errors": [], "info": []}
            
            # Phase 3: Manifest splitting
            split_output_dir = output_dir / "manifests"
            manifest_split_stats = self._split_manifest(manifest_path, split_output_dir)
            
            # Phase 4: Feature extraction (if enabled)
            feature_extraction_stats = {}
            if self.config.enable_feature_extraction:
                feature_extraction_stats = self._extract_vector_features(
                    split_output_dir, output_dir, pipeline_config
                )
            
            # Phase 5: Integration testing (if enabled)
            if self.config.run_integration_tests:
                test_report = self.integration_tester.run_full_integration_test(
                    manifest_path, dataset_config_path
                )
                if test_report['summary']['success_rate'] < 0.9:
                    warnings.append(f"Integration test success rate: {test_report['summary']['success_rate']:.1%}")
            
            # Generate output paths
            output_paths = {
                "base_dir": str(output_dir),
                "train_manifest": str(split_output_dir / "train_manifest.json"),
                "test_manifest": str(split_output_dir / "test_manifest.json"),
                "logs_dir": str(logs_dir),
                "config_file": str(output_dir / "pipeline_config.json") if pipeline_config else None
            }
            
            if self.config.enable_feature_extraction:
                output_paths.update({
                    "train_features": str(output_dir / "features" / "train_features.npz"),
                    "test_features": str(output_dir / "features" / "test_features.npz"),
                    "kernel_dicts": str(output_dir / "features" / "kernel_dicts.npz")
                })
            
            # Export configuration if available
            if pipeline_config:
                self.config_mapper.export_config(pipeline_config, Path(output_paths["config_file"]))
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Create success results
            results = BridgeResults(
                success=True,
                execution_time_seconds=execution_time,
                manifest_split_stats=manifest_split_stats,
                validation_reports=validation_reports,
                config_validation=config_validation,
                feature_extraction_stats=feature_extraction_stats,
                error_summary=self.error_handler.get_error_summary(),
                output_paths=output_paths,
                warnings=warnings
            )
            
            # Export detailed logs if configured
            if self.config.export_detailed_logs:
                self._export_detailed_logs(output_dir, results)
            
            self.error_handler.logger.info(f"Pipeline completed successfully in {execution_time:.1f}s")
            return results
            
        except Exception as e:
            # Handle pipeline failure
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.error_handler.handle_error(
                "integration_bridge",
                e,
                ErrorSeverity.CRITICAL,
                {"manifest_path": str(manifest_path), "output_dir": str(output_dir)}
            )
            
            # Create failure results
            results = BridgeResults(
                success=False,
                execution_time_seconds=execution_time,
                manifest_split_stats={},
                validation_reports={},
                config_validation={"errors": [str(e)], "warnings": warnings, "info": []},
                feature_extraction_stats={},
                error_summary=self.error_handler.get_error_summary(),
                output_paths={},
                warnings=warnings
            )
            
            # Still export error logs
            if self.config.export_detailed_logs:
                self._export_detailed_logs(output_dir, results)
            
            if self.config.strict_error_handling:
                raise e
            
            return results
    
    def _run_initial_validation(self, manifest_path: Path) -> Dict[str, Any]:
        """Run initial validation of manifest and associated files."""
        validation_reports = {}
        
        # NPZ validation
        if self.config.validate_npz_files:
            validation_reports['npz_validation'] = self.npz_validator.validate_manifest_diagrams(manifest_path)
        
        # Path validation  
        if self.config.validate_paths:
            validation_reports['path_validation'] = self.path_resolver.validate_manifest_paths(manifest_path)
        
        return validation_reports
    
    def _setup_pipeline_configuration(self, 
                                    dataset_config_path: Path,
                                    output_dir: Path,
                                    manifest_path: Path) -> PipelineConfig:
        """Setup and validate pipeline configuration."""
        # Create pipeline configuration
        pipeline_config = self.config_mapper.create_pipeline_config(
            dataset_config_path, 
            output_dir,
            manifest_path=manifest_path
        )
        
        return pipeline_config
    
    def _split_manifest(self, manifest_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Split manifest with error handling."""
        try:
            split_stats = self.manifest_splitter.split_manifest(
                manifest_path, 
                output_dir,
                train_name="train_manifest.json",
                test_name="test_manifest.json"
            )
            
            return split_stats
            
        except Exception as e:
            self.error_handler.handle_error(
                "manifest_splitting",
                e,
                ErrorSeverity.HIGH,
                {"manifest_path": str(manifest_path), "output_dir": str(output_dir)}
            )
            raise e
    
    def _extract_vector_features(self, 
                               manifest_dir: Path,
                               output_dir: Path, 
                               pipeline_config: Optional[PipelineConfig]) -> Dict[str, Any]:
        """Extract vector stack features from split manifests."""
        features_dir = output_dir / "features"
        features_dir.mkdir(exist_ok=True)
        
        stats = {
            "train_samples_processed": 0,
            "test_samples_processed": 0,
            "feature_vector_dimension": 0,
            "extraction_success_rate": 0.0
        }
        
        try:
            # Use default vector stack config if pipeline config not available
            if pipeline_config:
                vs_config = pipeline_config.vector_stack
            else:
                vs_config = VectorStackConfig(random_seed=self.config.random_seed)
            
            # Load train manifest
            train_manifest_path = manifest_dir / "train_manifest.json"
            test_manifest_path = manifest_dir / "test_manifest.json"
            
            if not train_manifest_path.exists():
                raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")
            
            with open(train_manifest_path, 'r') as f:
                train_manifest = json.load(f)
            
            # Process limited number of samples for demonstration
            train_entries = train_manifest['entries'][:self.config.max_samples_for_testing]
            
            # Load training diagrams for kernel dictionary preparation
            train_diagrams_by_dim = {dim: [] for dim in vs_config.homology_dims}
            train_features = []
            train_labels = []
            
            successful_loads = 0
            
            for entry in train_entries:
                try:
                    # Resolve diagram path
                    if 'diagram_path' in entry:
                        diagram_path = Path(entry['diagram_path'])
                    else:
                        diagram_path = train_manifest_path.parent.parent / "diagrams" / entry['file']
                    
                    # Load diagrams with error handling
                    diagrams = self.error_handler.safe_diagram_loading(
                        diagram_path,
                        [f"dgm_H{dim}" for dim in vs_config.homology_dims],
                        "feature_extraction"
                    )
                    
                    # Collect for kernel dictionary
                    for dim in vs_config.homology_dims:
                        if dim in diagrams:
                            train_diagrams_by_dim[dim].append(diagrams[dim])
                    
                    successful_loads += 1
                    train_labels.append(entry.get('label', 0))
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        "feature_extraction",
                        e,
                        ErrorSeverity.MEDIUM,
                        {"entry": entry}
                    )
            
            if successful_loads == 0:
                raise ValueError("No diagrams successfully loaded for feature extraction")
            
            # Prepare kernel dictionaries
            kernel_dicts = prepare_kernel_dictionaries(
                train_diagrams_by_dim, 
                vs_config,
                out_path=features_dir / "kernel_dicts.npz"
            )
            
            # Extract features for training samples
            norm_stats = None
            for i, entry in enumerate(train_entries[:successful_loads]):
                try:
                    if 'diagram_path' in entry:
                        diagram_path = Path(entry['diagram_path'])
                    else:
                        diagram_path = train_manifest_path.parent.parent / "diagrams" / entry['file']
                    
                    diagrams = self.error_handler.safe_diagram_loading(
                        diagram_path,
                        [f"dgm_H{dim}" for dim in vs_config.homology_dims],
                        "feature_extraction"
                    )
                    
                    vec, norm_stats, spans = build_vector_stack(
                        diagrams, vs_config, kernel_dicts, 
                        norm_stats=norm_stats, fit_norm=(norm_stats is None)
                    )
                    
                    train_features.append(vec)
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        "feature_extraction",
                        e,
                        ErrorSeverity.MEDIUM,
                        {"entry_index": i}
                    )
            
            # Save training features
            if train_features:
                X_train = np.vstack(train_features)
                y_train = np.array(train_labels[:len(train_features)])
                
                np.savez_compressed(
                    features_dir / "train_features.npz",
                    X=X_train,
                    y=y_train
                )
                
                stats['train_samples_processed'] = len(train_features)
                stats['feature_vector_dimension'] = X_train.shape[1]
            
            # Process test samples if test manifest exists
            if test_manifest_path.exists():
                with open(test_manifest_path, 'r') as f:
                    test_manifest = json.load(f)
                
                test_entries = test_manifest['entries'][:self.config.max_samples_for_testing//2]
                test_features = []
                test_labels = []
                
                for entry in test_entries:
                    try:
                        if 'diagram_path' in entry:
                            diagram_path = Path(entry['diagram_path'])
                        else:
                            diagram_path = test_manifest_path.parent.parent / "diagrams" / entry['file']
                        
                        diagrams = self.error_handler.safe_diagram_loading(
                            diagram_path,
                            [f"dgm_H{dim}" for dim in vs_config.homology_dims],
                            "feature_extraction"
                        )
                        
                        vec, _, _ = build_vector_stack(
                            diagrams, vs_config, kernel_dicts,
                            norm_stats=norm_stats, fit_norm=False
                        )
                        
                        test_features.append(vec)
                        test_labels.append(entry.get('label', 0))
                        
                    except Exception as e:
                        self.error_handler.handle_error(
                            "feature_extraction",
                            e,
                            ErrorSeverity.MEDIUM,
                            {"test_entry": entry}
                        )
                
                # Save test features
                if test_features:
                    X_test = np.vstack(test_features)
                    y_test = np.array(test_labels)
                    
                    np.savez_compressed(
                        features_dir / "test_features.npz",
                        X=X_test,
                        y=y_test
                    )
                    
                    stats['test_samples_processed'] = len(test_features)
            
            # Calculate success rate
            total_attempted = len(train_entries) + (len(test_entries) if test_manifest_path.exists() else 0)
            total_successful = stats['train_samples_processed'] + stats['test_samples_processed']
            stats['extraction_success_rate'] = total_successful / total_attempted if total_attempted > 0 else 0.0
            
        except Exception as e:
            self.error_handler.handle_error(
                "feature_extraction",
                e,
                ErrorSeverity.HIGH,
                {"features_dir": str(features_dir)}
            )
            raise e
        
        return stats
    
    def _export_detailed_logs(self, output_dir: Path, results: BridgeResults):
        """Export detailed execution logs."""
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Export bridge results
        with open(logs_dir / "bridge_results.json", 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Export error log
        self.error_handler.export_error_log(logs_dir / "error_log.json")
        
        # Export configuration
        with open(logs_dir / "bridge_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)


def run_integration_bridge(manifest_path: Union[str, Path],
                          output_dir: Union[str, Path],
                          dataset_config_path: Optional[Union[str, Path]] = None,
                          config: Optional[BridgeConfig] = None) -> BridgeResults:
    """Convenience function to run complete integration bridge.
    
    Parameters
    ----------
    manifest_path : str | Path
        Path to existing diagram manifest
    output_dir : str | Path
        Output directory for results
    dataset_config_path : str | Path, optional
        Path to dataset configuration
    config : BridgeConfig, optional
        Bridge configuration
        
    Returns
    -------
    BridgeResults
        Complete pipeline results
    """
    bridge = IntegrationBridge(config)
    return bridge.run_complete_pipeline(manifest_path, dataset_config_path, output_dir)


if __name__ == "__main__":
    # Command-line interface
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python integration_bridge.py <manifest_path> <output_dir> [dataset_config_path]")
        print("  manifest_path: Path to existing diagram manifest JSON")
        print("  output_dir: Directory for pipeline outputs")
        print("  dataset_config_path: Optional dataset configuration YAML")
        sys.exit(1)
    
    manifest_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    dataset_config_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Run integration bridge
    config = BridgeConfig(
        run_integration_tests=False,  # Skip expensive tests for CLI usage
        max_samples_for_testing=50    # Limit for faster execution
    )
    
    print(f"Starting TDA vector stack integration bridge...")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")
    if dataset_config_path:
        print(f"Dataset config: {dataset_config_path}")
    
    try:
        results = run_integration_bridge(manifest_path, output_dir, dataset_config_path, config)
        
        print(f"\n=== Integration Bridge Results ===")
        print(f"Success: {results.success}")
        print(f"Execution time: {results.execution_time_seconds:.1f}s")
        
        if results.success:
            print(f"Train samples: {results.manifest_split_stats.get('counts', {}).get('train', 0)}")
            print(f"Test samples: {results.manifest_split_stats.get('counts', {}).get('test', 0)}")
            
            if results.feature_extraction_stats:
                print(f"Features extracted: {results.feature_extraction_stats.get('train_samples_processed', 0)} train, "
                      f"{results.feature_extraction_stats.get('test_samples_processed', 0)} test")
                print(f"Feature dimension: {results.feature_extraction_stats.get('feature_vector_dimension', 0)}")
        
        if results.warnings:
            print(f"\nWarnings ({len(results.warnings)}):")
            for warning in results.warnings[:5]:
                print(f"  - {warning}")
        
        error_count = results.error_summary.get('total_errors', 0)
        if error_count > 0:
            print(f"\nTotal errors handled: {error_count}")
        
        print(f"\nOutput directory: {output_dir}")
        print("Integration bridge completed successfully!")
        
    except Exception as e:
        print(f"\nIntegration bridge failed: {e}")
        sys.exit(1)