"""Error handling and graceful degradation for TDA pipeline integration.

Provides robust error handling with fallback strategies to ensure the pipeline
can continue operating even when individual components fail or data is malformed.

Key Features:
- Graceful degradation strategies for missing/corrupt data
- Comprehensive error logging and reporting
- Fallback mechanisms for common failure modes
- Recovery strategies for partial pipeline failures
- Performance monitoring and resource management
- Context-aware error messages with troubleshooting guidance
"""
from __future__ import annotations
import logging
import traceback
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
from enum import Enum

# Type variable for generic error handling
T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"           # Minor issues that don't affect results
    MEDIUM = "medium"     # Issues that may affect quality but not functionality
    HIGH = "high"         # Significant issues affecting functionality
    CRITICAL = "critical" # Issues that prevent pipeline execution


class RecoveryStrategy(Enum):
    """Available recovery strategies for error handling."""
    SKIP = "skip"                    # Skip the problematic item/operation
    DEFAULT = "default"              # Use default/fallback value
    INTERPOLATE = "interpolate"      # Interpolate from neighboring data
    RETRY = "retry"                  # Retry the operation
    PARTIAL = "partial"              # Use partial results
    FAIL_GRACEFULLY = "fail_gracefully"  # Fail with informative error


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: str
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str]
    recovery_strategy: Optional[RecoveryStrategy]
    recovery_successful: bool


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior."""
    max_errors_per_component: int = 100  # Maximum errors to track per component
    enable_recovery_logging: bool = True  # Log recovery attempts
    raise_on_critical: bool = True       # Raise exception on critical errors
    enable_stack_traces: bool = True     # Include stack traces in error records
    log_level: str = "INFO"              # Logging level
    error_log_file: Optional[str] = None # File path for error logging
    enable_performance_monitoring: bool = False  # Track performance metrics


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, config: ErrorHandlingConfig = None):
        self.config = config or ErrorHandlingConfig()
        self.error_records: List[ErrorRecord] = []
        self.component_error_counts: Dict[str, int] = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("tda_pipeline_errors")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler if configured
            if self.config.error_log_file:
                file_handler = logging.FileHandler(self.config.error_log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def handle_error(self, 
                    component: str,
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    recovery_strategy: Optional[RecoveryStrategy] = None) -> ErrorRecord:
        """Handle an error with appropriate logging and recovery.
        
        Parameters
        ----------
        component : str
            Name of the component where error occurred
        error : Exception
            The exception that was raised
        severity : ErrorSeverity
            Severity level of the error
        context : Dict[str, Any], optional
            Additional context information
        recovery_strategy : RecoveryStrategy, optional
            Strategy used for recovery
            
        Returns
        -------
        ErrorRecord
            Record of the error and handling
        """
        context = context or {}
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            component=component,
            error_type=type(error).__name__,
            severity=severity,
            message=str(error),
            context=context,
            stack_trace=traceback.format_exc() if self.config.enable_stack_traces else None,
            recovery_strategy=recovery_strategy,
            recovery_successful=False  # Will be updated if recovery succeeds
        )
        
        # Update error counts
        self.component_error_counts[component] = self.component_error_counts.get(component, 0) + 1
        
        # Add to records (with rotation if needed)
        self.error_records.append(error_record)
        if len(self.error_records) > self.config.max_errors_per_component:
            self.error_records = self.error_records[-self.config.max_errors_per_component:]
        
        # Log based on severity
        log_message = f"[{component}] {severity.value.upper()}: {error.args[0] if error.args else str(error)}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            if self.config.raise_on_critical:
                raise error
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        return error_record
    
    def with_error_handling(self, 
                           component: str, 
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL_GRACEFULLY,
                           fallback_value: Any = None):
        """Decorator for automatic error handling.
        
        Parameters
        ----------
        component : str
            Component name for error tracking
        severity : ErrorSeverity
            Default severity level
        recovery_strategy : RecoveryStrategy
            Strategy for error recovery
        fallback_value : Any
            Value to return if recovery strategy is DEFAULT
        """
        def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_record = self.handle_error(
                        component=component,
                        error=e,
                        severity=severity,
                        context={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys())
                        },
                        recovery_strategy=recovery_strategy
                    )
                    
                    # Apply recovery strategy
                    if recovery_strategy == RecoveryStrategy.DEFAULT:
                        error_record.recovery_successful = True
                        return fallback_value
                    elif recovery_strategy == RecoveryStrategy.SKIP:
                        error_record.recovery_successful = True
                        return None
                    else:
                        # For other strategies, re-raise or return None
                        if severity == ErrorSeverity.CRITICAL:
                            raise e
                        return None
            
            return wrapper
        return decorator
    
    def safe_file_load(self, 
                      file_path: Path, 
                      component: str,
                      loader_func: Callable[[Path], T],
                      fallback_value: Optional[T] = None) -> Optional[T]:
        """Safely load a file with error handling.
        
        Parameters
        ----------
        file_path : Path
            Path to file to load
        component : str
            Component name for error tracking
        loader_func : Callable
            Function to use for loading the file
        fallback_value : T, optional
            Value to return if loading fails
            
        Returns
        -------
        T | None
            Loaded data or fallback value
        """
        try:
            if not file_path.exists():
                self.handle_error(
                    component=component,
                    error=FileNotFoundError(f"File not found: {file_path}"),
                    severity=ErrorSeverity.HIGH,
                    context={"file_path": str(file_path)},
                    recovery_strategy=RecoveryStrategy.DEFAULT
                )
                return fallback_value
            
            return loader_func(file_path)
            
        except Exception as e:
            self.handle_error(
                component=component,
                error=e,
                severity=ErrorSeverity.HIGH,
                context={"file_path": str(file_path)},
                recovery_strategy=RecoveryStrategy.DEFAULT
            )
            return fallback_value
    
    def safe_manifest_processing(self, 
                                manifest_entries: List[Dict[str, Any]],
                                processor_func: Callable[[Dict[str, Any]], T],
                                component: str) -> Tuple[List[T], List[Dict[str, Any]]]:
        """Process manifest entries with error handling.
        
        Parameters
        ----------
        manifest_entries : List[Dict[str, Any]]
            Manifest entries to process
        processor_func : Callable
            Function to process each entry
        component : str
            Component name for error tracking
            
        Returns
        -------
        Tuple[List[T], List[Dict[str, Any]]]
            Successfully processed items and failed entries
        """
        successful_results = []
        failed_entries = []
        
        for i, entry in enumerate(manifest_entries):
            try:
                result = processor_func(entry)
                successful_results.append(result)
            except Exception as e:
                self.handle_error(
                    component=component,
                    error=e,
                    severity=ErrorSeverity.MEDIUM,
                    context={
                        "entry_index": i,
                        "entry_keys": list(entry.keys()) if isinstance(entry, dict) else None
                    },
                    recovery_strategy=RecoveryStrategy.SKIP
                )
                failed_entries.append(entry)
        
        # Log processing statistics
        success_rate = len(successful_results) / len(manifest_entries) if manifest_entries else 0
        self.logger.info(f"[{component}] Processed {len(successful_results)}/{len(manifest_entries)} entries "
                        f"(success rate: {success_rate:.1%})")
        
        return successful_results, failed_entries
    
    def safe_diagram_loading(self, 
                            npz_path: Path, 
                            expected_keys: List[str],
                            component: str) -> Dict[int, np.ndarray]:
        """Safely load persistence diagrams from NPZ file.
        
        Parameters
        ----------
        npz_path : Path
            Path to NPZ file
        expected_keys : List[str]
            Expected diagram keys (e.g., ['dgm_H0', 'dgm_H1'])
        component : str
            Component name for error tracking
            
        Returns
        -------
        Dict[int, np.ndarray]
            Loaded diagrams by dimension
        """
        diagrams: Dict[int, np.ndarray] = {}
        
        try:
            if not npz_path.exists():
                self.handle_error(
                    component=component,
                    error=FileNotFoundError(f"NPZ file not found: {npz_path}"),
                    severity=ErrorSeverity.HIGH,
                    context={"npz_path": str(npz_path)},
                    recovery_strategy=RecoveryStrategy.DEFAULT
                )
                # Return empty diagrams for expected dimensions
                for key in expected_keys:
                    if key.startswith('dgm_H'):
                        dim = int(key.split('H')[1])
                        diagrams[dim] = np.zeros((0, 2), dtype=np.float64)
                return diagrams
            
            npz_data = np.load(npz_path)
            
            for key in expected_keys:
                try:
                    if key in npz_data:
                        diagram = npz_data[key]
                        
                        # Validate diagram format
                        if diagram.ndim != 2 or (diagram.size > 0 and diagram.shape[1] != 2):
                            raise ValueError(f"Invalid diagram shape: {diagram.shape}")
                        
                        # Extract dimension
                        dim = int(key.split('H')[1])
                        diagrams[dim] = diagram
                    else:
                        # Missing key - create empty diagram
                        dim = int(key.split('H')[1])
                        diagrams[dim] = np.zeros((0, 2), dtype=np.float64)
                        self.logger.warning(f"[{component}] Missing diagram key {key} in {npz_path}")
                
                except Exception as e:
                    self.handle_error(
                        component=component,
                        error=e,
                        severity=ErrorSeverity.MEDIUM,
                        context={"npz_path": str(npz_path), "key": key},
                        recovery_strategy=RecoveryStrategy.DEFAULT
                    )
                    # Create empty diagram as fallback
                    if key.startswith('dgm_H'):
                        dim = int(key.split('H')[1])
                        diagrams[dim] = np.zeros((0, 2), dtype=np.float64)
            
            npz_data.close()
            
        except Exception as e:
            self.handle_error(
                component=component,
                error=e,
                severity=ErrorSeverity.HIGH,
                context={"npz_path": str(npz_path)},
                recovery_strategy=RecoveryStrategy.DEFAULT
            )
            # Return empty diagrams for all expected dimensions
            for key in expected_keys:
                if key.startswith('dgm_H'):
                    dim = int(key.split('H')[1])
                    diagrams[dim] = np.zeros((0, 2), dtype=np.float64)
        
        return diagrams
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors.
        
        Returns
        -------
        Dict[str, Any]
            Error summary with statistics and details
        """
        if not self.error_records:
            return {"total_errors": 0, "components": {}, "severity_counts": {}}
        
        # Count errors by severity
        severity_counts = {}
        for record in self.error_records:
            severity = record.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count errors by component
        component_details = {}
        for component, count in self.component_error_counts.items():
            component_errors = [r for r in self.error_records if r.component == component]
            component_details[component] = {
                "total_errors": count,
                "severity_breakdown": {},
                "recent_errors": [
                    {
                        "timestamp": r.timestamp,
                        "error_type": r.error_type,
                        "severity": r.severity.value,
                        "message": r.message[:100] + "..." if len(r.message) > 100 else r.message
                    }
                    for r in component_errors[-5:]  # Last 5 errors
                ]
            }
            
            for error in component_errors:
                severity = error.severity.value
                component_details[component]["severity_breakdown"][severity] = \
                    component_details[component]["severity_breakdown"].get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_records),
            "components": component_details,
            "severity_counts": severity_counts,
            "error_rate_by_component": {
                comp: count for comp, count in self.component_error_counts.items()
            }
        }
    
    def export_error_log(self, output_path: Path) -> None:
        """Export complete error log to JSON file.
        
        Parameters
        ----------
        output_path : Path
            Path to save error log JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        error_log = {
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "configuration": asdict(self.config),
            "summary": self.get_error_summary(),
            "detailed_records": [asdict(record) for record in self.error_records]
        }
        
        with open(output_path, 'w') as f:
            json.dump(error_log, f, indent=2, default=str)


# Global error handler instance for convenience
_global_error_handler: Optional[ErrorHandler] = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def safe_operation(component: str, 
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  fallback_value: Any = None):
    """Decorator for safe operations using global error handler."""
    return get_global_error_handler().with_error_handling(
        component=component,
        severity=severity,
        recovery_strategy=RecoveryStrategy.DEFAULT,
        fallback_value=fallback_value
    )


# Convenience functions using global handler
def safe_file_load(file_path: Path, loader_func: Callable, component: str = "file_io"):
    """Safely load file using global error handler."""
    return get_global_error_handler().safe_file_load(file_path, component, loader_func)


def safe_diagram_load(npz_path: Path, expected_dims: List[int], component: str = "diagram_loading"):
    """Safely load diagrams using global error handler."""
    expected_keys = [f"dgm_H{dim}" for dim in expected_dims]
    return get_global_error_handler().safe_diagram_loading(npz_path, expected_keys, component)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python error_handler.py <test_mode>")
        print("  test_mode: 'demo' to run demonstration")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "demo":
        # Demonstration of error handling
        config = ErrorHandlingConfig(error_log_file="error_demo.log")
        handler = ErrorHandler(config)
        
        # Simulate various error scenarios
        try:
            raise FileNotFoundError("Demo file not found")
        except Exception as e:
            handler.handle_error("demo_component", e, ErrorSeverity.MEDIUM)
        
        try:
            raise ValueError("Demo validation error")
        except Exception as e:
            handler.handle_error("demo_component", e, ErrorSeverity.HIGH)
        
        # Test safe file loading
        result = handler.safe_file_load(
            Path("nonexistent_file.json"),
            "file_demo",
            lambda p: json.load(p.open()),
            fallback_value={"fallback": True}
        )
        
        print("Demo file load result:", result)
        
        # Print error summary
        summary = handler.get_error_summary()
        print("\nError Summary:")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Components affected: {list(summary['components'].keys())}")
        
        # Export error log
        handler.export_error_log(Path("demo_error_log.json"))
        print("Error log exported to demo_error_log.json")