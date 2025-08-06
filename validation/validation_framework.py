#!/usr/bin/env python3
"""
Comprehensive Validation Framework
Prevents result discrepancies and ensures detailed evidence capture
"""
import sys
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
from contextlib import redirect_stdout, redirect_stderr
import io
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ValidationFramework:
    """
    Comprehensive validation system that captures everything needed for verification
    """
    
    def __init__(self, experiment_name: str, random_seed: int = 42):
        self.experiment_name = experiment_name
        self.random_seed = random_seed
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create validation directory structure
        self.base_dir = f"validation/{experiment_name}"
        self.results_dir = f"{self.base_dir}/results_{self.timestamp}"
        self.plots_dir = f"{self.base_dir}/plots_{self.timestamp}"
        self.data_dir = f"{self.base_dir}/data_{self.timestamp}"
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)  
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Storage for all results
        self.console_output = []
        self.metrics = {}
        self.plots = {}
        self.raw_data = {}
        self.validation_passed = False
    
    def capture_console_output(self):
        """Context manager to capture all console output"""
        class OutputCapture:
            def __init__(self, validator):
                self.validator = validator
                self.stdout_capture = io.StringIO()
                self.stderr_capture = io.StringIO()
            
            def __enter__(self):
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr
                sys.stdout = self.stdout_capture
                sys.stderr = self.stderr_capture
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore original stdout/stderr
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
                
                # Capture output
                stdout_content = self.stdout_capture.getvalue()
                stderr_content = self.stderr_capture.getvalue()
                
                # Save and display
                self.validator.console_output.append({
                    'timestamp': datetime.now().isoformat(),
                    'stdout': stdout_content,
                    'stderr': stderr_content
                })
                
                # Print to user (they need to see it)
                if stdout_content:
                    print(stdout_content, end='')
                if stderr_content:
                    print(stderr_content, end='', file=sys.stderr)
        
        return OutputCapture(self)
    
    def validate_classification_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray = None, 
                                      class_names: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive classification validation with mandatory visualizations
        """
        print(f"\n🧪 COMPREHENSIVE VALIDATION: {self.experiment_name.upper()}")
        print("=" * 80)
        print(f"Timestamp: {datetime.now()}")
        print(f"Random Seed: {self.random_seed}")
        print(f"Validation ID: {self.timestamp}")
        print("=" * 80)
        
        # Calculate all metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store raw data for audit
        self.raw_data['y_true'] = y_true.tolist()
        self.raw_data['y_pred'] = y_pred.tolist()
        if y_pred_proba is not None:
            self.raw_data['y_pred_proba'] = y_pred_proba.tolist()
        
        # Generate comprehensive metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'random_seed': self.random_seed,
            'timestamp': self.timestamp,
            'experiment_name': self.experiment_name
        }
        
        print(f"\n📊 CORE METRICS")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"F1-Score: {f1:.3f} ({f1*100:.1f}%)")
        
        # Detailed classification report
        print(f"\n📈 DETAILED CLASSIFICATION REPORT")
        class_report = classification_report(y_true, y_pred, digits=3)
        print(class_report)
        
        # Store classification report
        self.raw_data['classification_report'] = class_report
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n🔍 CONFUSION MATRIX")
        print("Raw Confusion Matrix:")
        print(cm)
        
        # Generate mandatory visualizations
        self._generate_confusion_matrix_plot(cm, class_names)
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            self._generate_roc_curve(y_true, y_pred_proba)
            self._generate_precision_recall_curve(y_true, y_pred_proba)
        
        self._generate_prediction_distribution(y_true, y_pred)
        
        # Save all data
        self._save_validation_package()
        
        print(f"\n✅ VALIDATION COMPLETE")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📊 Plots saved to: {self.plots_dir}")
        print(f"💾 Raw data saved to: {self.data_dir}")
        
        return self.metrics
    
    def _generate_confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str] = None):
        """Generate and save confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {self.experiment_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        if class_names:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        # Add validation info
        fig.text(0.02, 0.02, f'Validation: {self.timestamp} | Seed: {self.random_seed}', 
                fontsize=8, alpha=0.7)
        
        # Save plot
        plot_path = f"{self.plots_dir}/confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots['confusion_matrix'] = plot_path
        print(f"📊 Confusion matrix saved: {plot_path}")
    
    def _generate_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Generate and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {self.experiment_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add validation info
        fig.text(0.02, 0.02, f'Validation: {self.timestamp} | Seed: {self.random_seed}', 
                fontsize=8, alpha=0.7)
        
        plot_path = f"{self.plots_dir}/roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots['roc_curve'] = plot_path
        self.metrics['roc_auc'] = float(roc_auc)
        print(f"📊 ROC curve saved: {plot_path} (AUC: {roc_auc:.3f})")
    
    def _generate_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Generate and save Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='red', lw=2,
               label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {self.experiment_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        # Add validation info
        fig.text(0.02, 0.02, f'Validation: {self.timestamp} | Seed: {self.random_seed}', 
                fontsize=8, alpha=0.7)
        
        plot_path = f"{self.plots_dir}/precision_recall_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots['precision_recall_curve'] = plot_path
        self.metrics['pr_auc'] = float(pr_auc)
        print(f"📊 Precision-Recall curve saved: {plot_path} (AUC: {pr_auc:.3f})")
    
    def _generate_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Generate prediction distribution analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True label distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar(unique_true, counts_true, alpha=0.7, color='blue', label='True')
        ax1.set_title('True Label Distribution', fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.legend()
        
        # Predicted label distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar(unique_pred, counts_pred, alpha=0.7, color='red', label='Predicted')
        ax2.set_title('Predicted Label Distribution', fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.legend()
        
        plt.suptitle(f'Label Distributions - {self.experiment_name}', fontsize=14, fontweight='bold')
        
        # Add validation info
        fig.text(0.02, 0.02, f'Validation: {self.timestamp} | Seed: {self.random_seed}', 
                fontsize=8, alpha=0.7)
        
        plot_path = f"{self.plots_dir}/prediction_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots['prediction_distribution'] = plot_path
        print(f"📊 Prediction distribution saved: {plot_path}")
    
    def _save_validation_package(self):
        """Save complete validation package for audit"""
        
        # Save console output
        console_file = f"{self.results_dir}/console_output.txt"
        with open(console_file, 'w') as f:
            f.write(f"Validation Console Output - {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Random Seed: {self.random_seed}\n")
            f.write("=" * 80 + "\n\n")
            
            for entry in self.console_output:
                f.write(f"[{entry['timestamp']}]\n")
                if entry['stdout']:
                    f.write("STDOUT:\n")
                    f.write(entry['stdout'])
                if entry['stderr']:
                    f.write("STDERR:\n")
                    f.write(entry['stderr'])
                f.write("\n" + "-" * 40 + "\n")
        
        # Save metrics
        metrics_file = f"{self.results_dir}/metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save raw data
        raw_data_file = f"{self.data_dir}/raw_data.json"
        with open(raw_data_file, 'w') as f:
            json.dump(self.raw_data, f, indent=2)
        
        # Save validation summary
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'random_seed': self.random_seed,
            'validation_passed': self.validation_passed,
            'metrics': self.metrics,
            'plots': self.plots,
            'files': {
                'console_output': console_file,
                'metrics': metrics_file,
                'raw_data': raw_data_file
            }
        }
        
        summary_file = f"{self.results_dir}/validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 VALIDATION PACKAGE SAVED")
        print(f"📋 Summary: {summary_file}")
        print(f"📊 Metrics: {metrics_file}")
        print(f"📝 Console: {console_file}")
        print(f"💾 Raw Data: {raw_data_file}")
    
    def verify_claim(self, claimed_f1: float, tolerance: float = 0.05) -> bool:
        """
        Verify a performance claim against validation results
        """
        if 'f1_score' not in self.metrics:
            raise ValueError("Must run validation before verifying claims")
        
        validated_f1 = self.metrics['f1_score']
        difference = abs(validated_f1 - claimed_f1)
        
        print(f"\n🔍 CLAIM VERIFICATION")
        print(f"Claimed F1-Score: {claimed_f1:.3f}")
        print(f"Validated F1-Score: {validated_f1:.3f}")
        print(f"Difference: {difference:.3f}")
        print(f"Tolerance: {tolerance:.3f}")
        
        if difference <= tolerance:
            print("✅ CLAIM VERIFIED: Within acceptable tolerance")
            self.validation_passed = True
            return True
        else:
            print("❌ CLAIM REJECTED: Outside acceptable tolerance")
            self.validation_passed = False
            return False

# Decorator to require validation
def require_validation(func):
    """Decorator that requires validation before allowing result reporting"""
    def wrapper(*args, **kwargs):
        if 'validator' not in kwargs or not isinstance(kwargs['validator'], ValidationFramework):
            raise ValueError("All result reporting requires ValidationFramework instance")
        
        validator = kwargs['validator']
        if not validator.validation_passed:
            raise ValueError("Cannot report results - validation not passed")
        
        return func(*args, **kwargs)
    return wrapper

@require_validation
def report_validated_results(experiment_name: str, validator: ValidationFramework) -> str:
    """
    Generate standard validated results report
    """
    metrics = validator.metrics
    plots = validator.plots
    
    report = f"""
# Validated Results: {experiment_name}

## Performance Metrics (Validated)
- **F1-Score**: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)
- **Accuracy**: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)
- **Validation Status**: ✅ VERIFIED
- **Random Seed**: {metrics['random_seed']}
- **Timestamp**: {metrics['timestamp']}

## Evidence Package
"""
    
    for plot_name, plot_path in plots.items():
        report += f"- **{plot_name.replace('_', ' ').title()}**: {plot_path}\n"
    
    report += f"""
## Validation Confirmation
✅ Results independently verified with ValidationFramework
✅ All required visualizations generated  
✅ Raw data and console output captured
✅ Reproducible with seed={metrics['random_seed']}

## Audit Trail
- Validation ID: {validator.timestamp}
- Results Directory: {validator.results_dir}
- Plots Directory: {validator.plots_dir}
- Raw Data Directory: {validator.data_dir}
"""
    
    return report

if __name__ == "__main__":
    # Example usage
    print("ValidationFramework - Preventing Result Discrepancies")
    print("Usage: Import and use ValidationFramework in all experiments")