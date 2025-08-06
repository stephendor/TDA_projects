#!/usr/bin/env python3
"""
Real Data Only Validator
Comprehensive validation using only real CIC-IDS2017 data with automatic graph generation
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Import our algorithms
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.algorithms.hybrid.hybrid_multiscale_graph_tda import HybridTDAAnalyzer
from validation.validation_framework import ValidationFramework

class RealDataValidator:
    """
    Comprehensive validator using only real CIC-IDS2017 data
    Automatically saves results and opens summary graphs
    """
    
    def __init__(self, output_dir=None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"validation/real_data_validation/{self.timestamp}"
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/plots").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/data").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/results").mkdir(exist_ok=True)
        
        self.results = {}
        self.plots_created = []
        
        print("🔍 REAL DATA VALIDATOR INITIALIZED")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Output directory: {self.output_dir}")
        print("NO SYNTHETIC DATA - REAL CIC-IDS2017 ONLY")
        print("=" * 60)
    
    def load_cic_data(self, attack_type='infiltration', max_samples=10000):
        """Load real CIC-IDS2017 data for specified attack type"""
        
        attack_files = {
            'infiltration': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'ddos': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'portscan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'webattacks': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'botnet': 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
        }
        
        if attack_type not in attack_files:
            raise ValueError(f"Attack type {attack_type} not supported. Available: {list(attack_files.keys())}")
        
        file_path = Path(f"data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /{attack_files[attack_type]}")
        
        print(f"\\n📊 LOADING REAL CIC-IDS2017 DATA: {attack_type.upper()}")
        print(f"File: {attack_files[attack_type]}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"CIC-IDS2017 file not found: {file_path}")
        
        # Load real data
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        print(f"✅ Raw data loaded: {len(df):,} samples")
        
        # Analyze attack distribution
        attack_mask = df['Label'] != 'BENIGN'
        attack_count = attack_mask.sum()
        benign_count = len(df) - attack_count
        
        print(f"   Attack samples: {attack_count:,} ({attack_count/len(df)*100:.2f}%)")
        print(f"   Benign samples: {benign_count:,} ({benign_count/len(df)*100:.2f}%)")
        
        if attack_count == 0:
            print(f"⚠️ No attacks found in {attack_type} data")
        
        # Sample if dataset is too large
        if len(df) > max_samples:
            print(f"   Sampling {max_samples:,} samples from {len(df):,} total")
            
            # Ensure we keep all attacks if possible
            attacks = df[attack_mask]
            benign = df[~attack_mask]
            
            if len(attacks) > 0:
                n_benign = min(len(benign), max_samples - len(attacks))
                benign_sample = benign.sample(n=n_benign, random_state=42)
                df = pd.concat([attacks, benign_sample]).sample(frac=1, random_state=42)
            else:
                df = df.sample(n=max_samples, random_state=42)
            
            print(f"   Final sample: {len(df):,} samples")
            print(f"   Attack rate: {(df['Label'] != 'BENIGN').mean()*100:.2f}%")
        
        # Save dataset info
        dataset_info = {
            'attack_type': attack_type,
            'file_path': str(file_path),
            'total_samples': len(df),
            'attack_samples': int((df['Label'] != 'BENIGN').sum()),
            'attack_rate': float((df['Label'] != 'BENIGN').mean()),
            'columns': list(df.columns),
            'timestamp': self.timestamp
        }
        
        with open(f"{self.output_dir}/data/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return df
    
    def validate_hybrid_baseline(self, df):
        """Validate hybrid TDA method - our 70.6% F1-score baseline"""
        
        print("\\n🔬 VALIDATING HYBRID TDA BASELINE (70.6% F1-SCORE TARGET)")
        print("-" * 60)
        
        # Initialize validation framework
        validator = ValidationFramework("real_data_hybrid_baseline", random_seed=42)
        
        try:
            with validator.capture_console_output():
                # Initialize hybrid analyzer
                analyzer = HybridTDAAnalyzer()
                
                # Extract features using real data only
                features, labels = analyzer.extract_hybrid_features(df)
                
                if features is None:
                    raise ValueError("Feature extraction failed - no synthetic fallback allowed")
                
                print(f"✅ Features extracted: {features.shape}")
                print(f"   Attack sequences: {labels.sum()}")
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=42, stratify=labels
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create ensemble (same as original baseline)
                rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                           random_state=42, class_weight='balanced')
                rf2 = RandomForestClassifier(n_estimators=150, max_depth=15, 
                                           random_state=123, class_weight='balanced')
                lr = LogisticRegression(C=0.1, random_state=42, 
                                      class_weight='balanced', max_iter=1000)
                
                ensemble = VotingClassifier(
                    estimators=[('rf1', rf1), ('rf2', rf2), ('lr', lr)],
                    voting='soft'
                )
                
                # Train and predict
                ensemble.fit(X_train_scaled, y_train)
                y_pred = ensemble.predict(X_test_scaled)
                y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_test)) == 2 else None
            
            # Validate results with framework
            results = validator.validate_classification_results(y_test, y_pred, y_pred_proba)
            
            # Verify against 70.6% baseline
            baseline_f1 = 0.706
            claim_verified = validator.verify_claim(baseline_f1, tolerance=0.05)
            
            results['baseline_comparison'] = {
                'baseline_f1': baseline_f1,
                'actual_f1': results['f1_score'],
                'difference': results['f1_score'] - baseline_f1,
                'claim_verified': claim_verified
            }
            
            # Save detailed results
            with open(f"{self.output_dir}/results/hybrid_baseline_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            self.results['hybrid_baseline'] = results
            
            # Generate comparison plot
            self._create_baseline_comparison_plot(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Hybrid baseline validation failed: {e}")
            error_result = {'status': 'ERROR', 'error': str(e), 'f1_score': 0.0}
            self.results['hybrid_baseline'] = error_result
            return error_result
    
    def validate_multi_attack_performance(self):
        """Validate performance across multiple attack types"""
        
        print("\\n🎯 MULTI-ATTACK VALIDATION")
        print("-" * 60)
        
        attack_types = ['infiltration', 'ddos', 'portscan', 'webattacks']
        multi_results = {}
        
        for attack_type in attack_types:
            print(f"\\n📊 Testing on {attack_type.upper()} attacks...")
            
            try:
                # Load data for this attack type
                df = self.load_cic_data(attack_type, max_samples=5000)
                
                # Quick validation
                result = self.validate_hybrid_baseline(df)
                multi_results[attack_type] = {
                    'f1_score': result['f1_score'],
                    'accuracy': result['accuracy'],
                    'status': result.get('status', 'COMPLETED')
                }
                
                print(f"   {attack_type}: F1={result['f1_score']:.3f}")
                
            except Exception as e:
                print(f"   ❌ {attack_type} failed: {e}")
                multi_results[attack_type] = {'status': 'ERROR', 'error': str(e)}
        
        # Save multi-attack results
        with open(f"{self.output_dir}/results/multi_attack_results.json", 'w') as f:
            json.dump(multi_results, f, indent=2)
        
        # Generate multi-attack comparison plot
        self._create_multi_attack_plot(multi_results)
        
        self.results['multi_attack'] = multi_results
        return multi_results
    
    def _create_baseline_comparison_plot(self, results):
        """Create baseline comparison visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance comparison
        baseline_f1 = results['baseline_comparison']['baseline_f1']
        actual_f1 = results['baseline_comparison']['actual_f1']
        
        ax1.bar(['Baseline\\n(Historical)', 'Current\\n(Real Data)'], 
               [baseline_f1, actual_f1], 
               color=['lightblue', 'darkblue' if actual_f1 >= baseline_f1 else 'red'])
        ax1.set_ylabel('F1-Score')
        ax1.set_title('Baseline vs Current Performance', fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate([baseline_f1, actual_f1]):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Metrics overview
        metrics = ['accuracy', 'f1_score']
        if 'precision' in results:
            metrics.extend(['precision', 'recall'])
        
        values = [results.get(m, 0) for m in metrics]
        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in values]
        
        ax2.bar(metrics, values, color=colors)
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Metrics', fontweight='bold')
        ax2.set_ylim(0, 1)
        
        for i, v in enumerate(values):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 3. ROC Curve (if available)
        if 'roc_auc' in results:
            ax3.text(0.5, 0.7, f"ROC AUC: {results['roc_auc']:.3f}", 
                    ha='center', fontsize=16, fontweight='bold',
                    transform=ax3.transAxes)
            ax3.text(0.5, 0.3, "See separate ROC curve\\nin validation output", 
                    ha='center', fontsize=12,
                    transform=ax3.transAxes)
        ax3.set_title('ROC Performance', fontweight='bold')
        ax3.axis('off')
        
        # 4. Status summary
        status = results.get('status', 'UNKNOWN')
        verified = results['baseline_comparison']['claim_verified']
        
        status_color = 'green' if verified else 'red'
        ax4.text(0.5, 0.7, f"Status: {status}", 
                ha='center', fontsize=16, fontweight='bold',
                color=status_color, transform=ax4.transAxes)
        
        ax4.text(0.5, 0.5, f"Baseline Verified: {'✅' if verified else '❌'}", 
                ha='center', fontsize=14, transform=ax4.transAxes)
        
        ax4.text(0.5, 0.3, f"Real Data Only: ✅", 
                ha='center', fontsize=14, color='green',
                transform=ax4.transAxes)
        
        ax4.set_title('Validation Status', fontweight='bold')
        ax4.axis('off')
        
        # Overall title
        difference = results['baseline_comparison']['difference']
        diff_text = f"{difference:+.3f}" if difference != 0 else "±0.000"
        plt.suptitle(f'Real Data Validation Summary\\nF1-Score: {actual_f1:.3f} (Baseline: {baseline_f1:.3f}, Δ{diff_text})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save and add to plots list
        plot_path = f"{self.output_dir}/plots/baseline_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.plots_created.append(plot_path)
        
        print(f"✅ Baseline comparison plot saved: {plot_path}")
        
        return plot_path
    
    def _create_multi_attack_plot(self, multi_results):
        """Create multi-attack performance visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract successful results
        successful_results = {k: v for k, v in multi_results.items() 
                            if v.get('status') != 'ERROR' and 'f1_score' in v}
        
        if successful_results:
            # 1. F1-Score comparison
            attack_types = list(successful_results.keys())
            f1_scores = [successful_results[k]['f1_score'] for k in attack_types]
            
            colors = ['green' if f >= 0.7 else 'orange' if f >= 0.5 else 'red' for f in f1_scores]
            
            bars = ax1.bar(attack_types, f1_scores, color=colors)
            ax1.set_ylabel('F1-Score')
            ax1.set_title('Performance Across Attack Types', fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', fontweight='bold')
            
            # Add baseline line
            ax1.axhline(y=0.706, color='blue', linestyle='--', alpha=0.7, 
                       label='Baseline (70.6%)')
            ax1.legend()
            
            # 2. Success rate
            total_tests = len(multi_results)
            successful_tests = len(successful_results)
            failed_tests = total_tests - successful_tests
            
            ax2.pie([successful_tests, failed_tests], 
                   labels=[f'Successful\\n({successful_tests})', f'Failed\\n({failed_tests})'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%',
                   startangle=90)
            ax2.set_title('Multi-Attack Test Results', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No successful validations', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=16)
            ax1.set_title('Performance Across Attack Types', fontweight='bold')
            
            ax2.text(0.5, 0.5, 'All tests failed', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=16, color='red')
            ax2.set_title('Multi-Attack Test Results', fontweight='bold')
        
        plt.suptitle(f'Multi-Attack Validation Results\\nReal CIC-IDS2017 Data Only', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save and add to plots list
        plot_path = f"{self.output_dir}/plots/multi_attack_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.plots_created.append(plot_path)
        
        print(f"✅ Multi-attack comparison plot saved: {plot_path}")
        
        return plot_path
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        print("\\n📋 GENERATING SUMMARY REPORT")
        print("-" * 60)
        
        summary = {
            'timestamp': self.timestamp,
            'validation_type': 'REAL_DATA_ONLY',
            'synthetic_data_usage': 'FORBIDDEN',
            'baseline_target': '70.6% F1-Score',
            'results': self.results,
            'plots_created': self.plots_created,
            'output_directory': self.output_dir
        }
        
        # Save summary
        summary_path = f"{self.output_dir}/validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        markdown_report = self._create_markdown_report(summary)
        
        report_path = f"{self.output_dir}/VALIDATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        
        print(f"✅ Summary report saved: {summary_path}")
        print(f"✅ Markdown report saved: {report_path}")
        
        return summary_path, report_path
    
    def _create_markdown_report(self, summary):
        """Create markdown validation report"""
        
        report = f"""# Real Data Validation Report

## Validation Summary
- **Timestamp**: {summary['timestamp']}
- **Data Source**: Real CIC-IDS2017 Only
- **Synthetic Data**: ❌ FORBIDDEN
- **Baseline Target**: 70.6% F1-Score

## Results

### Hybrid Baseline Validation
"""
        
        if 'hybrid_baseline' in summary['results']:
            result = summary['results']['hybrid_baseline']
            if 'f1_score' in result:
                status = "✅ PASSED" if result.get('baseline_comparison', {}).get('claim_verified', False) else "❌ FAILED"
                f1_score = result['f1_score']
                baseline_f1 = result.get('baseline_comparison', {}).get('baseline_f1', 0.706)
                difference = f1_score - baseline_f1
                
                report += f"""
- **Status**: {status}
- **F1-Score**: {f1_score:.3f}
- **Baseline**: {baseline_f1:.3f}
- **Difference**: {difference:+.3f}
- **Accuracy**: {result.get('accuracy', 0):.3f}
"""
            else:
                report += f"""
- **Status**: ❌ ERROR
- **Error**: {result.get('error', 'Unknown error')}
"""
        
        if 'multi_attack' in summary['results']:
            report += "\\n### Multi-Attack Validation\\n"
            multi_results = summary['results']['multi_attack']
            
            for attack_type, result in multi_results.items():
                if 'f1_score' in result:
                    report += f"- **{attack_type.upper()}**: F1={result['f1_score']:.3f}\\n"
                else:
                    report += f"- **{attack_type.upper()}**: ERROR\\n"
        
        report += f"""

## Generated Plots
"""
        for plot in summary['plots_created']:
            plot_name = Path(plot).name
            report += f"- {plot_name}\\n"
        
        report += f"""

## Output Directory
All validation artifacts saved to: `{summary['output_directory']}`

## Validation Integrity
- ✅ Real CIC-IDS2017 data only
- ✅ No synthetic data generation  
- ✅ Reproducible results (seed=42)
- ✅ Complete audit trail
"""
        
        return report
    
    def auto_open_plots(self):
        """Automatically open generated plots"""
        
        if not self.plots_created:
            print("⚠️ No plots to open")
            return
        
        print(f"\\n🖼️ OPENING {len(self.plots_created)} SUMMARY PLOTS")
        
        for plot_path in self.plots_created:
            try:
                if sys.platform.startswith('linux'):
                    subprocess.run(['xdg-open', plot_path], check=True)
                elif sys.platform.startswith('darwin'):  # macOS
                    subprocess.run(['open', plot_path], check=True)
                elif sys.platform.startswith('win'):
                    subprocess.run(['start', plot_path], shell=True, check=True)
                
                print(f"✅ Opened: {Path(plot_path).name}")
                
            except Exception as e:
                print(f"⚠️ Could not auto-open {Path(plot_path).name}: {e}")
                print(f"   Manual path: {plot_path}")


def main():
    """Main validation execution"""
    
    print("🔍 REAL DATA VALIDATION FRAMEWORK")
    print("=" * 60)
    print("NO SYNTHETIC DATA - CIC-IDS2017 ONLY")
    print("=" * 60)
    
    # Initialize validator
    validator = RealDataValidator()
    
    try:
        # Load real CIC-IDS2017 infiltration data
        df = validator.load_cic_data('infiltration', max_samples=8000)
        
        # Validate hybrid baseline
        baseline_results = validator.validate_hybrid_baseline(df)
        
        # Multi-attack validation
        multi_results = validator.validate_multi_attack_performance()
        
        # Generate summary report
        summary_path, report_path = validator.generate_summary_report()
        
        # Auto-open plots
        validator.auto_open_plots()
        
        print(f"\\n🎉 REAL DATA VALIDATION COMPLETE")
        print(f"📁 Results: {validator.output_dir}")
        print(f"📊 Plots: {len(validator.plots_created)} generated and opened")
        
        return validator.results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()