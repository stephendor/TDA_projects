"""Configuration parameter mapping for TDA pipeline integration.

Maps between different configuration formats used across the TDA system:
- Dataset YAML configs (CIC-IDS format) 
- Vector Stack configs (VectorStackConfig)
- Manifest metadata (diagram dumps)
- Baseline pipeline configs

Ensures parameter consistency and provides validation with helpful error messages.

Key Features:
- Bi-directional mapping between config formats
- Parameter validation with type checking
- Default value handling and inheritance
- Configuration diff reporting for troubleshooting
- Batch configuration processing for multiple datasets
"""
from __future__ import annotations
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict, fields
import numpy as np

# Import target config classes
from src.embeddings.vector_stack import VectorStackConfig


@dataclass
class DatasetConfig:
    """Standardized dataset configuration format."""
    name: str
    format: str
    root_path: str
    file_pattern: str
    time_column: str
    window_seconds: float
    window_overlap: float
    feature_columns: List[str]
    categorical_columns: List[str]
    max_vr_points: int
    witness_landmarks: int
    maxdim: int = 2
    random_seed: int = 42


@dataclass 
class PipelineConfig:
    """Unified pipeline configuration combining all components."""
    dataset: DatasetConfig
    vector_stack: VectorStackConfig
    diagram_dump_config: Dict[str, Any]
    output_paths: Dict[str, str]
    integration_params: Dict[str, Any]


class ConfigMapper:
    """Handles mapping between different configuration formats."""
    
    def __init__(self):
        self.default_mappings = self._setup_default_mappings()
    
    def _setup_default_mappings(self) -> Dict[str, Any]:
        """Setup default parameter mappings and conversions."""
        return {
            "homology_dims_mapping": {
                "maxdim_0": [0],
                "maxdim_1": [0, 1], 
                "maxdim_2": [0, 1, 2],
                "default": [0, 1]
            },
            "diagram_limits": {
                "max_points_per_dim_default": 300,
                "max_windows_default": 0  # 0 = unlimited
            },
            "feature_extraction_defaults": {
                "landscape_levels": 3,
                "landscape_resolutions": [100, 300],
                "image_grids": [[16, 16], [32, 32]],
                "betti_resolution": 300,
                "sw_num_angles": 32,
                "sw_resolution": 300,
                "kernel_sample_small": 10000,
                "kernel_sample_large": 20000
            }
        }
    
    def load_dataset_config(self, config_path: Path) -> DatasetConfig:
        """Load and parse dataset YAML configuration.
        
        Parameters
        ----------
        config_path : Path
            Path to dataset YAML configuration file
            
        Returns
        -------
        DatasetConfig
            Parsed and validated dataset configuration
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Extract and validate required fields
        try:
            dataset_config = DatasetConfig(
                name=raw_config['name'],
                format=raw_config['format'],
                root_path=raw_config['root_path'],
                file_pattern=raw_config['file_pattern'],
                time_column=raw_config['time_column'],
                window_seconds=raw_config['window']['seconds'],
                window_overlap=raw_config['window']['overlap'],
                feature_columns=raw_config['features']['numeric'],
                categorical_columns=raw_config['features'].get('categorical', []),
                max_vr_points=raw_config['limits']['max_vr_points'],
                witness_landmarks=raw_config['limits']['witness_landmarks'],
                maxdim=raw_config.get('maxdim', 2),
                random_seed=raw_config.get('random_seed', 42)
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in dataset config: {e}")
        
        return dataset_config
    
    def dataset_to_vector_stack_config(self, 
                                     dataset_config: DatasetConfig,
                                     overrides: Optional[Dict[str, Any]] = None) -> VectorStackConfig:
        """Convert dataset config to vector stack configuration.
        
        Parameters
        ----------
        dataset_config : DatasetConfig
            Source dataset configuration
        overrides : Dict[str, Any], optional
            Parameter overrides for vector stack config
            
        Returns
        -------
        VectorStackConfig
            Configured vector stack parameters
        """
        # Map homology dimensions based on maxdim
        homology_dims = self.default_mappings["homology_dims_mapping"].get(
            f"maxdim_{dataset_config.maxdim}",
            self.default_mappings["homology_dims_mapping"]["default"]
        )
        
        # Base vector stack configuration
        vs_config_params = {
            "homology_dims": tuple(homology_dims),
            "random_seed": dataset_config.random_seed,
            "max_points_per_block": dataset_config.max_vr_points,
            **self.default_mappings["feature_extraction_defaults"]
        }
        
        # Apply overrides if provided
        if overrides:
            vs_config_params.update(overrides)
        
        return VectorStackConfig(**vs_config_params)
    
    def manifest_to_diagram_config(self, manifest_path: Path) -> Dict[str, Any]:
        """Extract diagram dump configuration from manifest metadata.
        
        Parameters
        ----------
        manifest_path : Path
            Path to diagram manifest JSON file
            
        Returns
        -------
        Dict[str, Any]
            Diagram dump configuration parameters
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        config = manifest.get('config', {})
        
        # Standardize configuration keys
        diagram_config = {
            "max_points_per_dim": config.get("DIAGRAM_MAX_POINTS_PER_DIM", 
                                           self.default_mappings["diagram_limits"]["max_points_per_dim_default"]),
            "max_dump_windows": config.get("MAX_DUMP_WINDOWS",
                                         self.default_mappings["diagram_limits"]["max_windows_default"]),
            "schema_version": manifest.get("schema_version", 1),
            "total_windows": manifest.get("total_windows", 0),
            "created_utc": manifest.get("created_utc", ""),
        }
        
        return diagram_config
    
    def create_pipeline_config(self, 
                              dataset_config_path: Path,
                              output_dir: Path,
                              vector_stack_overrides: Optional[Dict[str, Any]] = None,
                              manifest_path: Optional[Path] = None) -> PipelineConfig:
        """Create unified pipeline configuration from components.
        
        Parameters
        ----------
        dataset_config_path : Path
            Path to dataset YAML configuration
        output_dir : Path
            Base output directory for pipeline results
        vector_stack_overrides : Dict[str, Any], optional
            Overrides for vector stack configuration
        manifest_path : Path, optional
            Existing diagram manifest (for config extraction)
            
        Returns
        -------
        PipelineConfig
            Complete pipeline configuration
        """
        # Load dataset configuration
        dataset_config = self.load_dataset_config(dataset_config_path)
        
        # Create vector stack configuration
        vector_stack_config = self.dataset_to_vector_stack_config(
            dataset_config, vector_stack_overrides
        )
        
        # Extract diagram dump config if manifest provided
        if manifest_path and manifest_path.exists():
            diagram_dump_config = self.manifest_to_diagram_config(manifest_path)
        else:
            diagram_dump_config = self.default_mappings["diagram_limits"].copy()
        
        # Define output paths
        output_paths = {
            "base_dir": str(output_dir),
            "train_manifest": str(output_dir / "train_manifest.json"),
            "test_manifest": str(output_dir / "test_manifest.json"), 
            "vector_features": str(output_dir / "vector_features"),
            "results": str(output_dir / "results"),
            "plots": str(output_dir / "plots"),
            "logs": str(output_dir / "logs")
        }
        
        # Integration parameters
        integration_params = {
            "temporal_split_ratio": 0.7,
            "validation_enabled": True,
            "path_resolution_strict": True,
            "error_handling_mode": "strict",
            "feature_normalization": True,
            "reproducible_splits": True
        }
        
        return PipelineConfig(
            dataset=dataset_config,
            vector_stack=vector_stack_config,
            diagram_dump_config=diagram_dump_config,
            output_paths=output_paths,
            integration_params=integration_params
        )
    
    def validate_config_compatibility(self, pipeline_config: PipelineConfig) -> Dict[str, Any]:
        """Validate compatibility between configuration components.
        
        Parameters
        ----------
        pipeline_config : PipelineConfig
            Complete pipeline configuration to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation report with warnings and errors
        """
        issues = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        dataset = pipeline_config.dataset
        vs_config = pipeline_config.vector_stack
        
        # Check homology dimension consistency
        max_vs_dim = max(vs_config.homology_dims) if vs_config.homology_dims else 0
        if max_vs_dim > dataset.maxdim:
            issues["errors"].append(
                f"Vector stack max homology dim ({max_vs_dim}) exceeds "
                f"dataset maxdim ({dataset.maxdim})"
            )
        
        # Check point count consistency
        if vs_config.max_points_per_block > dataset.max_vr_points:
            issues["warnings"].append(
                f"Vector stack max_points_per_block ({vs_config.max_points_per_block}) "
                f"exceeds dataset max_vr_points ({dataset.max_vr_points})"
            )
        
        # Check random seed consistency  
        if vs_config.random_seed != dataset.random_seed:
            issues["warnings"].append(
                f"Random seed mismatch: dataset ({dataset.random_seed}) vs "
                f"vector_stack ({vs_config.random_seed})"
            )
        
        # Path validation
        dataset_root = Path(dataset.root_path)
        if not dataset_root.exists():
            issues["errors"].append(f"Dataset root path does not exist: {dataset_root}")
        
        # Feature extraction parameter validation
        for resolution in vs_config.landscape_resolutions:
            if resolution < 10:
                issues["warnings"].append(f"Very low landscape resolution: {resolution}")
                
        for grid in vs_config.image_grids:
            if isinstance(grid, (list, tuple)) and len(grid) == 2:
                if grid[0] * grid[1] < 16:
                    issues["warnings"].append(f"Very small persistence image grid: {grid}")
        
        # Info about configuration
        issues["info"].append(f"Dataset: {dataset.name} ({dataset.format})")
        issues["info"].append(f"Homology dimensions: {vs_config.homology_dims}")
        issues["info"].append(f"Feature columns: {len(dataset.feature_columns)}")
        
        return issues
    
    def export_config(self, pipeline_config: PipelineConfig, output_path: Path) -> None:
        """Export pipeline configuration to JSON file.
        
        Parameters
        ----------
        pipeline_config : PipelineConfig
            Configuration to export
        output_path : Path
            Output JSON file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        config_dict = {
            "dataset": asdict(pipeline_config.dataset),
            "vector_stack": asdict(pipeline_config.vector_stack),
            "diagram_dump_config": pipeline_config.diagram_dump_config,
            "output_paths": pipeline_config.output_paths,
            "integration_params": pipeline_config.integration_params
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def load_pipeline_config(self, config_path: Path) -> PipelineConfig:
        """Load pipeline configuration from JSON file.
        
        Parameters
        ----------
        config_path : Path
            Path to exported pipeline configuration
            
        Returns
        -------
        PipelineConfig
            Loaded pipeline configuration
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct configuration objects
        dataset_config = DatasetConfig(**config_dict["dataset"])
        
        # Handle tuple conversion for vector stack
        vs_dict = config_dict["vector_stack"].copy()
        if "homology_dims" in vs_dict:
            vs_dict["homology_dims"] = tuple(vs_dict["homology_dims"])
        vector_stack_config = VectorStackConfig(**vs_dict)
        
        return PipelineConfig(
            dataset=dataset_config,
            vector_stack=vector_stack_config,
            diagram_dump_config=config_dict["diagram_dump_config"],
            output_paths=config_dict["output_paths"],
            integration_params=config_dict["integration_params"]
        )


def create_config_for_dataset(dataset_config_path: Union[str, Path], 
                             output_dir: Union[str, Path],
                             **overrides) -> PipelineConfig:
    """Convenience function to create pipeline config for a dataset.
    
    Parameters
    ----------
    dataset_config_path : str | Path
        Path to dataset YAML configuration
    output_dir : str | Path
        Output directory for pipeline results
    **overrides
        Additional configuration overrides
        
    Returns
    -------
    PipelineConfig
        Complete pipeline configuration
    """
    mapper = ConfigMapper()
    return mapper.create_pipeline_config(
        Path(dataset_config_path),
        Path(output_dir),
        vector_stack_overrides=overrides
    )


if __name__ == "__main__":
    # Example usage and validation
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python config_mapper.py <dataset_config.yaml> <output_dir> [export_path]")
        sys.exit(1)
    
    dataset_config_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    export_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Create and validate configuration
    mapper = ConfigMapper()
    pipeline_config = mapper.create_pipeline_config(dataset_config_path, output_dir)
    
    # Validate compatibility
    issues = mapper.validate_config_compatibility(pipeline_config)
    
    print(f"Pipeline configuration for: {pipeline_config.dataset.name}")
    print(f"Vector stack homology dims: {pipeline_config.vector_stack.homology_dims}")
    
    if issues["errors"]:
        print("\nErrors:")
        for error in issues["errors"]:
            print(f"  - {error}")
    
    if issues["warnings"]:
        print("\nWarnings:")  
        for warning in issues["warnings"]:
            print(f"  - {warning}")
    
    # Export if requested
    if export_path:
        mapper.export_config(pipeline_config, export_path)
        print(f"\nConfiguration exported to: {export_path}")