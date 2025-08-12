"""NPZ format validation for persistence diagram data integrity.

Validates NPZ files containing persistence diagrams against expected formats,
ensuring compatibility between existing dumps and vector stack processing.

Key Features:
- Schema validation for diagram NPZ files
- Key naming consistency checks (dgm_H0, dgm_H1, etc.)
- Data type and shape validation
- Finite value verification (birth/death pairs)
- Batch validation for multiple files
- Detailed error reporting and statistics
- Format migration assistance for legacy files
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from dataclasses import dataclass, asdict
import warnings


@dataclass
class NPZValidationConfig:
    """Configuration for NPZ validation behavior."""
    expected_homology_dims: List[int] = None  # Expected H0, H1, H2 etc.
    require_finite_values: bool = True  # Check for inf/nan in diagrams
    allow_empty_diagrams: bool = True  # Allow zero-point diagrams
    check_birth_death_order: bool = True  # Verify birth <= death
    max_diagram_size: Optional[int] = None  # Warn if diagrams are very large
    min_diagram_size: Optional[int] = None  # Warn if diagrams are unexpectedly small
    expected_dtype: str = "float64"  # Expected numpy dtype for diagram points
    
    def __post_init__(self):
        if self.expected_homology_dims is None:
            self.expected_homology_dims = [0, 1, 2]


@dataclass
class NPZValidationResult:
    """Results of NPZ file validation."""
    file_path: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    diagram_stats: Dict[str, Dict[str, Any]]


class NPZValidator:
    """Validates persistence diagram NPZ files."""
    
    def __init__(self, config: NPZValidationConfig = None):
        self.config = config or NPZValidationConfig()
    
    def validate_file(self, npz_path: Path) -> NPZValidationResult:
        """Validate a single NPZ file containing persistence diagrams.
        
        Parameters
        ----------
        npz_path : Path
            Path to NPZ file to validate
            
        Returns
        -------
        NPZValidationResult
            Detailed validation results
        """
        errors = []
        warnings = []
        info = {}
        diagram_stats = {}
        
        # Basic file existence check
        if not npz_path.exists():
            return NPZValidationResult(
                file_path=str(npz_path),
                is_valid=False,
                errors=[f"File does not exist: {npz_path}"],
                warnings=[],
                info={},
                diagram_stats={}
            )
        
        try:
            # Load NPZ file
            npz_data = np.load(npz_path)
            keys = list(npz_data.keys())
            info['keys'] = keys
            info['file_size_bytes'] = npz_path.stat().st_size
            
            # Check for expected diagram keys
            expected_keys = [f"dgm_H{dim}" for dim in self.config.expected_homology_dims]
            missing_keys = [key for key in expected_keys if key not in keys]
            unexpected_keys = [key for key in keys if not key.startswith('dgm_H')]
            
            if missing_keys:
                errors.append(f"Missing expected diagram keys: {missing_keys}")
            
            if unexpected_keys:
                warnings.append(f"Unexpected keys found: {unexpected_keys}")
            
            # Validate each diagram
            for key in keys:
                if key.startswith('dgm_H'):
                    try:
                        dim_str = key.split('H')[1]
                        dim = int(dim_str)
                    except (IndexError, ValueError):
                        errors.append(f"Invalid diagram key format: {key}")
                        continue
                    
                    diagram = npz_data[key]
                    stats = self._validate_diagram(diagram, key, dim)
                    
                    diagram_stats[key] = stats
                    errors.extend(stats.get('errors', []))
                    warnings.extend(stats.get('warnings', []))
            
            info['total_diagrams'] = len([k for k in keys if k.startswith('dgm_H')])
            
        except Exception as e:
            errors.append(f"Failed to load NPZ file: {str(e)}")
        finally:
            try:
                npz_data.close()
            except:
                pass
        
        is_valid = len(errors) == 0
        
        return NPZValidationResult(
            file_path=str(npz_path),
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            diagram_stats=diagram_stats
        )
    
    def _validate_diagram(self, diagram: np.ndarray, key: str, dim: int) -> Dict[str, Any]:
        """Validate individual persistence diagram array."""
        stats = {
            'shape': diagram.shape,
            'dtype': str(diagram.dtype),
            'size': diagram.size,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        # Check basic shape requirements
        if diagram.ndim != 2:
            stats['errors'].append(f"{key}: Expected 2D array, got {diagram.ndim}D")
            return stats
        
        if diagram.size > 0 and diagram.shape[1] != 2:
            stats['errors'].append(f"{key}: Expected 2 columns (birth, death), got {diagram.shape[1]}")
            return stats
        
        # Check data type
        if str(diagram.dtype) != self.config.expected_dtype:
            stats['warnings'].append(
                f"{key}: Expected dtype {self.config.expected_dtype}, got {diagram.dtype}"
            )
        
        # Handle empty diagrams
        if diagram.size == 0:
            if not self.config.allow_empty_diagrams:
                stats['errors'].append(f"{key}: Empty diagram not allowed")
            else:
                stats['properties']['empty'] = True
            return stats
        
        # Extract birth and death values
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        
        # Check for finite values
        if self.config.require_finite_values:
            birth_finite = np.isfinite(births)
            death_finite = np.isfinite(deaths)
            
            if not birth_finite.all():
                non_finite_count = (~birth_finite).sum()
                stats['errors'].append(f"{key}: {non_finite_count} non-finite birth values")
            
            # Deaths can be infinite for H0 dimension (connected components)
            if dim > 0 and not death_finite.all():
                non_finite_count = (~death_finite).sum()
                stats['warnings'].append(f"{key}: {non_finite_count} non-finite death values")
            elif dim == 0:
                # For H0, infinite deaths are expected for persistent components
                infinite_deaths = np.isinf(deaths).sum()
                stats['properties']['infinite_deaths'] = int(infinite_deaths)
        
        # Check birth <= death order
        if self.config.check_birth_death_order:
            finite_mask = np.isfinite(births) & np.isfinite(deaths)
            if finite_mask.any():
                finite_births = births[finite_mask]
                finite_deaths = deaths[finite_mask]
                violation_mask = finite_births > finite_deaths
                
                if violation_mask.any():
                    violation_count = violation_mask.sum()
                    stats['errors'].append(
                        f"{key}: {violation_count} points violate birth <= death order"
                    )
        
        # Size validation
        n_points = diagram.shape[0]
        
        if self.config.max_diagram_size and n_points > self.config.max_diagram_size:
            stats['warnings'].append(
                f"{key}: Large diagram with {n_points} points (max expected: {self.config.max_diagram_size})"
            )
        
        if self.config.min_diagram_size and n_points < self.config.min_diagram_size:
            stats['warnings'].append(
                f"{key}: Small diagram with {n_points} points (min expected: {self.config.min_diagram_size})"
            )
        
        # Statistical properties
        if n_points > 0:
            stats['properties'].update({
                'n_points': int(n_points),
                'birth_range': [float(births.min()), float(births.max())],
                'death_range': [float(deaths.min()), float(deaths.max())],
                'finite_points': int(np.isfinite(births).sum() and np.isfinite(deaths).sum())
            })
            
            # Lifetime statistics for finite points
            finite_mask = np.isfinite(births) & np.isfinite(deaths)
            if finite_mask.any():
                lifetimes = deaths[finite_mask] - births[finite_mask]
                stats['properties'].update({
                    'lifetime_mean': float(lifetimes.mean()),
                    'lifetime_std': float(lifetimes.std()),
                    'lifetime_range': [float(lifetimes.min()), float(lifetimes.max())]
                })
        
        return stats
    
    def validate_batch(self, npz_paths: List[Path]) -> Dict[str, Any]:
        """Validate multiple NPZ files and generate summary report.
        
        Parameters
        ----------
        npz_paths : List[Path]
            List of NPZ file paths to validate
            
        Returns
        -------
        Dict[str, Any]
            Batch validation summary report
        """
        results = []
        for path in npz_paths:
            result = self.validate_file(path)
            results.append(result)
        
        # Generate summary statistics
        total_files = len(results)
        valid_files = sum(1 for r in results if r.is_valid)
        
        all_errors = []
        all_warnings = []
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        # Count diagram statistics across all files
        dim_counts = {}
        total_diagrams = 0
        
        for result in results:
            for key in result.diagram_stats:
                if key.startswith('dgm_H'):
                    dim = key.split('H')[1]
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                    total_diagrams += 1
        
        summary = {
            'batch_validation_summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': total_files - valid_files,
                'success_rate': valid_files / total_files if total_files > 0 else 0.0,
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings),
                'total_diagrams': total_diagrams,
                'dimension_counts': dim_counts
            },
            'individual_results': [asdict(result) for result in results],
            'common_errors': self._count_common_issues(all_errors),
            'common_warnings': self._count_common_issues(all_warnings)
        }
        
        return summary
    
    def _count_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Count frequency of common error/warning patterns."""
        counts = {}
        for issue in issues:
            # Extract common patterns
            if "Missing expected diagram keys" in issue:
                counts['missing_keys'] = counts.get('missing_keys', 0) + 1
            elif "non-finite" in issue:
                counts['non_finite_values'] = counts.get('non_finite_values', 0) + 1
            elif "birth <= death order" in issue:
                counts['birth_death_order'] = counts.get('birth_death_order', 0) + 1
            elif "Empty diagram" in issue:
                counts['empty_diagrams'] = counts.get('empty_diagrams', 0) + 1
            else:
                counts['other'] = counts.get('other', 0) + 1
        
        return counts
    
    def validate_manifest_diagrams(self, manifest_path: Path) -> Dict[str, Any]:
        """Validate all diagrams referenced in a manifest file.
        
        Parameters
        ----------
        manifest_path : Path
            Path to diagram manifest JSON file
            
        Returns
        -------
        Dict[str, Any]
            Validation report for all diagrams in manifest
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        entries = manifest.get('entries', [])
        manifest_dir = manifest_path.parent
        
        # Extract NPZ file paths from manifest entries
        npz_paths = []
        for entry in entries:
            file_name = entry.get('file') or entry.get('diagram_path', '')
            if file_name:
                npz_path = manifest_dir / file_name
                npz_paths.append(npz_path)
        
        # Validate all files
        validation_report = self.validate_batch(npz_paths)
        
        # Add manifest-specific information
        validation_report['manifest_info'] = {
            'manifest_path': str(manifest_path),
            'total_entries': len(entries),
            'npz_files_found': len(npz_paths)
        }
        
        return validation_report


def validate_npz_file(npz_path: Union[str, Path], 
                     config: Optional[NPZValidationConfig] = None) -> NPZValidationResult:
    """Convenience function to validate a single NPZ file.
    
    Parameters
    ----------
    npz_path : str | Path
        Path to NPZ file
    config : NPZValidationConfig, optional
        Validation configuration (uses defaults if None)
        
    Returns
    -------
    NPZValidationResult
        Validation results
    """
    validator = NPZValidator(config)
    return validator.validate_file(Path(npz_path))


def validate_diagram_manifest(manifest_path: Union[str, Path],
                             config: Optional[NPZValidationConfig] = None) -> Dict[str, Any]:
    """Convenience function to validate all diagrams in a manifest.
    
    Parameters
    ----------
    manifest_path : str | Path
        Path to manifest JSON file
    config : NPZValidationConfig, optional
        Validation configuration
        
    Returns
    -------
    Dict[str, Any]
        Complete validation report
    """
    validator = NPZValidator(config)
    return validator.validate_manifest_diagrams(Path(manifest_path))


if __name__ == "__main__":
    # Command-line interface for validation
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python npz_validator.py <path> [output_report.json]")
        print("  path: NPZ file or manifest JSON to validate")
        print("  output_report: Optional JSON file to save validation report")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_report = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    validator = NPZValidator()
    
    if input_path.suffix == '.json':
        # Validate manifest
        report = validator.validate_manifest_diagrams(input_path)
        print(f"Manifest validation: {input_path}")
        print(f"Success rate: {report['batch_validation_summary']['success_rate']:.1%}")
        print(f"Valid files: {report['batch_validation_summary']['valid_files']}")
        print(f"Invalid files: {report['batch_validation_summary']['invalid_files']}")
        
        if report['batch_validation_summary']['total_errors'] > 0:
            print("\nCommon errors:")
            for error_type, count in report['common_errors'].items():
                print(f"  {error_type}: {count}")
    
    elif input_path.suffix == '.npz':
        # Validate single NPZ file
        result = validator.validate_file(input_path)
        print(f"NPZ validation: {input_path}")
        print(f"Valid: {result.is_valid}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        print(f"\nDiagrams found: {len(result.diagram_stats)}")
        for key, stats in result.diagram_stats.items():
            n_points = stats['properties'].get('n_points', 0)
            print(f"  {key}: {n_points} points")
        
        # Single file report
        report = asdict(result)
    
    else:
        print(f"Unsupported file type: {input_path.suffix}")
        sys.exit(1)
    
    # Save report if requested
    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nValidation report saved to: {output_report}")