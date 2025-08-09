"""Path resolution utilities for TDA pipeline integration.

Handles path resolution across different execution contexts, ensuring
compatibility between relative paths in manifests and absolute paths
required by vector stack processing.

Key Features:
- Context-aware path resolution (script directory, working directory)  
- Validation of file existence with helpful error messages
- Batch path resolution for manifest entries
- Cross-platform path handling
- Configurable search paths and fallback logic
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from dataclasses import dataclass


@dataclass 
class PathResolutionConfig:
    """Configuration for path resolution behavior."""
    search_paths: List[Path] = None  # Additional search directories
    validate_existence: bool = True  # Check if resolved paths exist
    create_missing_dirs: bool = False  # Create parent directories if missing
    allow_relative_fallback: bool = True  # Fall back to relative paths if absolute resolution fails
    
    def __post_init__(self):
        if self.search_paths is None:
            self.search_paths = []


class PathResolver:
    """Handles path resolution with multiple context strategies."""
    
    def __init__(self, config: PathResolutionConfig = None):
        self.config = config or PathResolutionConfig()
        
        # Standard project paths for fallback resolution
        self.project_root = self._find_project_root()
        self.validation_dir = self.project_root / "validation"
        self.diagrams_dir = self.validation_dir / "diagrams"
        
    def _find_project_root(self) -> Path:
        """Find project root by looking for characteristic files."""
        current = Path.cwd()
        
        # Look for project indicators
        indicators = ["setup.py", "pyproject.toml", "requirements.txt", ".git"]
        
        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return current
            current = current.parent
        
        # Fallback to current working directory
        return Path.cwd()
    
    def resolve_diagram_path(self, 
                           path_spec: Union[str, Path],
                           context_dir: Optional[Path] = None,
                           manifest_dir: Optional[Path] = None) -> Path:
        """Resolve diagram file path with multiple fallback strategies.
        
        Parameters
        ----------
        path_spec : str | Path
            Path specification from manifest or config
        context_dir : Path, optional
            Directory context for relative path resolution
        manifest_dir : Path, optional
            Directory containing the manifest (for relative paths)
            
        Returns
        -------
        Path
            Resolved absolute path
            
        Raises
        ------
        FileNotFoundError
            If path cannot be resolved and validation is enabled
        """
        path_spec = Path(path_spec)
        
        # If already absolute and exists, return as-is
        if path_spec.is_absolute():
            if not self.config.validate_existence or path_spec.exists():
                return path_spec
        
        # Resolution strategies in order of preference
        strategies = [
            ("manifest_relative", lambda: manifest_dir / path_spec if manifest_dir else None),
            ("context_relative", lambda: context_dir / path_spec if context_dir else None),
            ("diagrams_dir", lambda: self.diagrams_dir / path_spec.name),
            ("validation_relative", lambda: self.validation_dir / path_spec),
            ("project_relative", lambda: self.project_root / path_spec),
            ("cwd_relative", lambda: Path.cwd() / path_spec),
        ]
        
        # Add configured search paths
        for search_path in self.config.search_paths:
            strategies.append(
                (f"search_{search_path.name}", lambda sp=search_path: sp / path_spec)
            )
        
        # Try each strategy
        for strategy_name, resolver in strategies:
            try:
                candidate = resolver()
                if candidate is None:
                    continue
                    
                if not self.config.validate_existence or candidate.exists():
                    return candidate.resolve()
                    
            except Exception:
                continue
        
        # Final fallback strategies
        if self.config.allow_relative_fallback:
            # Try the original path as-is (might be relative to current working dir)
            if not self.config.validate_existence or path_spec.exists():
                return path_spec.resolve() if not path_spec.is_absolute() else path_spec
        
        # If validation is disabled, return best guess
        if not self.config.validate_existence:
            return (self.diagrams_dir / path_spec.name).resolve()
        
        # All strategies failed
        raise FileNotFoundError(
            f"Could not resolve diagram path: {path_spec}\n"
            f"Searched in: {[str(s[0]) for s in strategies]}\n"
            f"Project root: {self.project_root}\n"
            f"Diagrams dir: {self.diagrams_dir}"
        )
    
    def resolve_manifest_entries(self, 
                                entries: List[Dict[str, Any]],
                                manifest_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Resolve paths for all entries in a manifest.
        
        Parameters
        ----------
        entries : List[Dict[str, Any]]
            Manifest entries with 'file' or 'diagram_path' fields
        manifest_path : Path, optional
            Path to the manifest file (for relative resolution context)
            
        Returns
        -------
        List[Dict[str, Any]]
            Entries with resolved 'diagram_path' field added
        """
        manifest_dir = manifest_path.parent if manifest_path else None
        resolved_entries = []
        
        for entry in entries:
            resolved_entry = entry.copy()
            
            # Get path specification from entry
            path_spec = entry.get('diagram_path') or entry.get('file')
            
            if path_spec:
                try:
                    resolved_path = self.resolve_diagram_path(
                        path_spec, 
                        manifest_dir=manifest_dir
                    )
                    resolved_entry['diagram_path'] = str(resolved_path)
                    resolved_entry['path_resolved'] = True
                except FileNotFoundError as e:
                    if self.config.validate_existence:
                        raise e
                    else:
                        # Store unresolved path with warning flag
                        resolved_entry['diagram_path'] = str(path_spec)
                        resolved_entry['path_resolved'] = False
                        resolved_entry['resolution_warning'] = str(e)
            
            resolved_entries.append(resolved_entry)
        
        return resolved_entries
    
    def resolve_output_path(self, 
                           path_spec: Union[str, Path],
                           create_parents: bool = None) -> Path:
        """Resolve output path with parent directory creation.
        
        Parameters
        ----------
        path_spec : str | Path
            Output path specification
        create_parents : bool, optional
            Override config setting for parent directory creation
            
        Returns
        -------
        Path
            Resolved absolute path with parents created if requested
        """
        path_spec = Path(path_spec)
        
        # Resolve relative to project root for outputs
        if not path_spec.is_absolute():
            path_spec = self.project_root / path_spec
        
        # Create parent directories if requested
        should_create = create_parents if create_parents is not None else self.config.create_missing_dirs
        
        if should_create:
            path_spec.parent.mkdir(parents=True, exist_ok=True)
        
        return path_spec.resolve()
    
    def validate_manifest_paths(self, manifest_path: Path) -> Dict[str, Any]:
        """Validate all paths in a manifest file.
        
        Parameters
        ----------
        manifest_path : Path
            Path to manifest JSON file
            
        Returns
        -------
        Dict[str, Any]
            Validation report with statistics and issues
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        entries = manifest.get('entries', [])
        
        # Track validation results
        total_entries = len(entries)
        resolved_count = 0
        missing_files = []
        resolution_errors = []
        
        for i, entry in enumerate(entries):
            path_spec = entry.get('diagram_path') or entry.get('file')
            
            if not path_spec:
                resolution_errors.append({
                    'entry_index': i,
                    'error': 'No path specification found',
                    'entry': entry
                })
                continue
            
            try:
                resolved_path = self.resolve_diagram_path(
                    path_spec, 
                    manifest_dir=manifest_path.parent
                )
                resolved_count += 1
            except FileNotFoundError as e:
                missing_files.append({
                    'entry_index': i,
                    'path_spec': path_spec,
                    'error': str(e)
                })
        
        validation_report = {
            'manifest_path': str(manifest_path),
            'validation_timestamp': Path(__file__).stat().st_mtime,  # Use file mtime as simple timestamp
            'total_entries': total_entries,
            'resolved_count': resolved_count,
            'missing_count': len(missing_files),
            'error_count': len(resolution_errors),
            'success_rate': resolved_count / total_entries if total_entries > 0 else 0.0,
            'missing_files': missing_files[:10],  # Limit for brevity
            'resolution_errors': resolution_errors[:5],
            'search_paths': [str(p) for p in self.config.search_paths],
            'project_root': str(self.project_root),
            'diagrams_dir': str(self.diagrams_dir)
        }
        
        return validation_report


def resolve_diagram_path(path_spec: Union[str, Path], 
                        manifest_dir: Optional[Path] = None,
                        validate: bool = True) -> Path:
    """Convenience function for single path resolution.
    
    Parameters
    ----------
    path_spec : str | Path
        Path specification to resolve
    manifest_dir : Path, optional
        Directory containing manifest (for context)
    validate : bool
        Whether to validate path existence
        
    Returns
    -------
    Path
        Resolved absolute path
    """
    config = PathResolutionConfig(validate_existence=validate)
    resolver = PathResolver(config)
    return resolver.resolve_diagram_path(path_spec, manifest_dir=manifest_dir)


def batch_resolve_paths(entries: List[Dict[str, Any]], 
                       manifest_path: Optional[Path] = None,
                       validate: bool = True) -> List[Dict[str, Any]]:
    """Convenience function for batch path resolution.
    
    Parameters
    ----------
    entries : List[Dict[str, Any]]
        Manifest entries to process
    manifest_path : Path, optional
        Path to manifest file
    validate : bool
        Whether to validate path existence
        
    Returns
    -------
    List[Dict[str, Any]]
        Entries with resolved paths
    """
    config = PathResolutionConfig(validate_existence=validate)
    resolver = PathResolver(config)
    return resolver.resolve_manifest_entries(entries, manifest_path)


if __name__ == "__main__":
    # Example usage and validation
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python path_resolver.py <manifest_path> [validate]")
        print("       python path_resolver.py <diagram_path>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    validate = len(sys.argv) < 3 or sys.argv[2].lower() != 'false'
    
    resolver = PathResolver(PathResolutionConfig(validate_existence=validate))
    
    if path.suffix == '.json':
        # Validate manifest
        report = resolver.validate_manifest_paths(path)
        print(f"Validation report for {path}:")
        print(f"Success rate: {report['success_rate']:.1%}")
        print(f"Resolved: {report['resolved_count']}/{report['total_entries']}")
        if report['missing_count'] > 0:
            print(f"Missing files: {report['missing_count']}")
            for missing in report['missing_files'][:3]:
                print(f"  - {missing['path_spec']}")
    else:
        # Resolve single path
        try:
            resolved = resolver.resolve_diagram_path(path)
            print(f"Resolved path: {resolved}")
            print(f"Exists: {resolved.exists()}")
        except FileNotFoundError as e:
            print(f"Resolution failed: {e}")