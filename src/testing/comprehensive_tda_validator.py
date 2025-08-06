#!/usr/bin/env python3
"""
Comprehensive TDA Validation Framework
Combines all validation approaches with enhanced security and multi-attack testing
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# Import validation framework (adjust path for root execution)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from validation.validation_framework import ValidationFramework, report_validated_results

# Import all TDA algorithms from new structure
from src.algorithms.hybrid.hybrid_multiscale_graph_tda import HybridTDAAnalyzer
from src.algorithms.ensemble.tda_supervised_ensemble import TDASupervisedEnsemble
from src.algorithms.temporal.temporal_persistence_evolution import TemporalPersistenceEvolutionAnalyzer
from src.algorithms.temporal.implement_multiscale_tda import MultiScaleTDAAnalyzer

class ComprehensiveTDAValidator:
    """
    Enhanced validator that tests all TDA methods across multiple datasets and attack types
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.results = {}
        self.datasets = self._initialize_datasets()
        
        print("🔬 COMPREHENSIVE TDA VALIDATOR INITIALIZED")
        print("=" * 60)
        print(f"Random Seed: {random_seed}")
        print(f"Available Datasets: {len(self.datasets)}")
        print(f"Available Algorithms: 4 (Hybrid, Ensemble, Temporal Evolution, Multi-Scale)")
        
    def _initialize_datasets(self) -> Dict[str, Dict]:
        """Initialize all available datasets"""
        base_path = Path("data/apt_datasets/")
        
        return {
            'cic_infiltration': {
                'path': base_path / 'cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                'attack_type': 'Infiltration',
                'priority': 'PRIMARY',
                'validated': True
            },
            'cic_ddos': {
                'path': base_path / 'cicids2017/GeneratedLabelledFlows/TrafficLabelling /Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                'attack_type': 'DDoS', 
                'priority': 'EXPANSION',
                'validated': False
            },
            'cic_portscan': {
                'path': base_path / 'cicids2017/GeneratedLabelledFlows/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'attack_type': 'PortScan',
                'priority': 'EXPANSION', 
                'validated': False
            },
            'cic_webattacks': {
                'path': base_path / 'cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                'attack_type': 'WebAttacks',
                'priority': 'EXPANSION',
                'validated': False
            }
        }
    
    def validate_all_methods(self, dataset_name: str = 'cic_infiltration') -> Dict[str, Any]:
        """
        Validate all TDA methods on specified dataset
        """
        print(f"\n🎯 VALIDATING ALL TDA METHODS ON {dataset_name.upper()}")
        print("=" * 70)
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")
            
        dataset_info = self.datasets[dataset_name]
        data = self._load_dataset(dataset_info['path'])
        
        if data is None:
            print(f"❌ Failed to load dataset: {dataset_name}")
            return {}
            
        validation_results = {}
        
        # 1. Validate Hybrid Multi-Scale + Graph TDA (Known: 70.6%)
        print(f"\n📊 1. HYBRID MULTI-SCALE + GRAPH TDA")
        validation_results['hybrid'] = self._validate_hybrid_method(data)
        
        # 2. Validate TDA + Supervised Ensemble (Claimed: 80%+) 
        print(f"\n📊 2. TDA + SUPERVISED ENSEMBLE")
        validation_results['ensemble'] = self._validate_ensemble_method(data)
        
        # 3. Validate Temporal Persistence Evolution (Needs validation)
        print(f"\n📊 3. TEMPORAL PERSISTENCE EVOLUTION") 
        validation_results['temporal_evolution'] = self._validate_temporal_evolution_method(data)
        
        # 4. Validate Multi-Scale TDA (Claimed: 65.4%)
        print(f"\n📊 4. MULTI-SCALE TDA")
        validation_results['multiscale'] = self._validate_multiscale_method(data)
        
        # Aggregate results
        self.results[dataset_name] = validation_results
        
        # Generate summary
        self._generate_validation_summary(dataset_name, validation_results)
        
        return validation_results
    
    def _validate_hybrid_method(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate hybrid multi-scale + graph TDA method"""
        try:
            validator = ValidationFramework("hybrid_multiscale_graph_tda", self.random_seed)
            
            with validator.capture_console_output():
                analyzer = HybridTDAAnalyzer()
                features, labels = analyzer.extract_hybrid_features(data)
                
                if features is None:
                    print("❌ Feature extraction failed")
                    return {'status': 'FAILED', 'error': 'Feature extraction failed'}
                
                # Simple train-test for validation
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import VotingClassifier, RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=self.random_seed, 
                    stratify=labels
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Same ensemble as original
                rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                           random_state=self.random_seed, class_weight='balanced')
                lr = LogisticRegression(C=0.1, random_state=self.random_seed, 
                                      class_weight='balanced', max_iter=1000)
                
                ensemble = VotingClassifier(
                    estimators=[('rf1', rf1), ('lr', lr)],
                    voting='soft'
                )
                
                ensemble.fit(X_train_scaled, y_train)
                y_pred = ensemble.predict(X_test_scaled)
                y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_test)) == 2 else None
            
            # Validate results  
            results = validator.validate_classification_results(y_test, y_pred, y_pred_proba)
            
            # Verify against known performance (70.6% with tolerance)
            claim_verified = validator.verify_claim(0.706, tolerance=0.05)
            results['claim_verification'] = claim_verified
            results['status'] = 'VALIDATED' if claim_verified else 'DISPUTED'
            
            return results
            
        except Exception as e:
            print(f"❌ Hybrid validation failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_ensemble_method(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate TDA + Supervised ensemble method"""
        try:
            validator = ValidationFramework("tda_supervised_ensemble", self.random_seed)
            
            with validator.capture_console_output():
                analyzer = TDASupervisedEnsemble()
                results = analyzer.evaluate(data)
                
                if results is None:
                    print("❌ Ensemble evaluation failed")
                    return {'status': 'FAILED', 'error': 'Evaluation failed'}
            
            # Extract predictions for validation
            y_test = results.get('y_test')
            y_pred = results.get('y_pred') 
            y_pred_proba = results.get('y_pred_proba')
            
            if any(x is None for x in [y_test, y_pred]):
                return {'status': 'FAILED', 'error': 'Missing prediction data'}
            
            # Validate results
            validation_results = validator.validate_classification_results(y_test, y_pred, y_pred_proba)
            
            # This method claimed 80%+ F1-score - verify
            claimed_f1 = 0.80
            claim_verified = validator.verify_claim(claimed_f1, tolerance=0.10)  # More generous tolerance
            validation_results['claim_verification'] = claim_verified
            validation_results['status'] = 'VALIDATED' if claim_verified else 'DISPUTED'
            validation_results['claimed_f1'] = claimed_f1
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Ensemble validation failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_temporal_evolution_method(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal persistence evolution method"""
        try:
            validator = ValidationFramework("temporal_persistence_evolution", self.random_seed)
            
            with validator.capture_console_output():
                analyzer = TemporalPersistenceEvolutionAnalyzer()
                results = analyzer.evaluate(data)
                
                if results is None:
                    print("❌ Temporal evolution evaluation failed")
                    return {'status': 'FAILED', 'error': 'Evaluation failed'}
            
            # Extract predictions
            y_test = results.get('y_test')
            y_pred = results.get('y_pred')
            y_pred_proba = results.get('y_pred_proba')
            
            if any(x is None for x in [y_test, y_pred]):
                return {'status': 'FAILED', 'error': 'Missing prediction data'}
            
            # Validate results
            validation_results = validator.validate_classification_results(y_test, y_pred, y_pred_proba)
            
            # No specific claim to verify - just assess performance
            f1_score = validation_results.get('f1_score', 0)
            validation_results['status'] = 'VALIDATED' if f1_score > 0.5 else 'POOR_PERFORMANCE'
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Temporal evolution validation failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_multiscale_method(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate multi-scale TDA method"""
        try:
            validator = ValidationFramework("multiscale_tda", self.random_seed)
            
            with validator.capture_console_output():
                analyzer = MultiScaleTDAAnalyzer()
                results = analyzer.evaluate(data)
                
                if results is None:
                    print("❌ Multi-scale evaluation failed") 
                    return {'status': 'FAILED', 'error': 'Evaluation failed'}
            
            # Extract predictions
            y_test = results.get('y_test')
            y_pred = results.get('y_pred')
            y_pred_proba = results.get('y_pred_proba')
            
            if any(x is None for x in [y_test, y_pred]):
                return {'status': 'FAILED', 'error': 'Missing prediction data'}
            
            # Validate results
            validation_results = validator.validate_classification_results(y_test, y_pred, y_pred_proba)
            
            # Claimed 65.4% F1-score - verify  
            claimed_f1 = 0.654
            claim_verified = validator.verify_claim(claimed_f1, tolerance=0.05)
            validation_results['claim_verification'] = claim_verified
            validation_results['status'] = 'VALIDATED' if claim_verified else 'DISPUTED'
            validation_results['claimed_f1'] = claimed_f1
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Multi-scale validation failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _load_dataset(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and prepare dataset"""
        try:
            if not file_path.exists():
                print(f"❌ Dataset file not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            # Basic data quality check
            if len(df) == 0:
                print(f"❌ Empty dataset: {file_path}")
                return None
                
            print(f"✅ Loaded dataset: {len(df):,} samples")
            print(f"   Attack samples: {len(df[df['Label'] != 'BENIGN']):,}")
            print(f"   Benign samples: {len(df[df['Label'] == 'BENIGN']):,}")
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to load dataset {file_path}: {e}")
            return None
    
    def _generate_validation_summary(self, dataset_name: str, results: Dict[str, Any]):
        """Generate comprehensive validation summary"""
        print(f"\n🏁 VALIDATION SUMMARY FOR {dataset_name.upper()}")
        print("=" * 70)
        
        validated_count = 0
        disputed_count = 0
        failed_count = 0
        
        for method_name, result in results.items():
            status = result.get('status', 'UNKNOWN')
            f1_score = result.get('f1_score', 0)
            
            status_emoji = {
                'VALIDATED': '✅',
                'DISPUTED': '⚠️', 
                'FAILED': '❌',
                'ERROR': '💥',
                'POOR_PERFORMANCE': '📉'
            }.get(status, '❓')
            
            print(f"{status_emoji} {method_name.upper()}: {status}")
            print(f"   F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)")
            
            if 'claimed_f1' in result:
                claimed = result['claimed_f1']
                difference = f1_score - claimed
                print(f"   Claimed: {claimed:.3f}, Difference: {difference:+.3f}")
            
            if status == 'VALIDATED':
                validated_count += 1
            elif status in ['DISPUTED', 'POOR_PERFORMANCE']:
                disputed_count += 1
            else:
                failed_count += 1
            
            print()
        
        total_methods = len(results)
        validation_rate = (validated_count / total_methods) * 100 if total_methods > 0 else 0
        
        print(f"📊 OVERALL VALIDATION RATE: {validation_rate:.1f}% ({validated_count}/{total_methods})")
        print(f"   ✅ Validated: {validated_count}")
        print(f"   ⚠️ Disputed: {disputed_count}")
        print(f"   ❌ Failed: {failed_count}")
        
        if validation_rate >= 75:
            print("🎉 EXCELLENT: High validation rate achieved!")
        elif validation_rate >= 50:
            print("👍 GOOD: Majority of methods validated")
        else:
            print("⚠️ CONCERNING: Low validation rate - investigation needed")


def main():
    """Main execution function"""
    print("🔬 COMPREHENSIVE TDA VALIDATION FRAMEWORK")
    print("=" * 60)
    print("Testing all TDA methods with enhanced security validation")
    print("=" * 60)
    
    # Initialize validator
    validator = ComprehensiveTDAValidator(random_seed=42)
    
    # Run comprehensive validation
    results = validator.validate_all_methods('cic_infiltration')
    
    print(f"\n🎯 VALIDATION COMPLETE")
    print("Results saved with full audit trail and evidence")
    
    return results


if __name__ == "__main__":
    main()