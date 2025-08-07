import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def run_detector_test(detector_class, detector_name, X, y, test_size=0.3, random_state=42, **kwargs):
    """Generalized function to test a detector model."""
    print(f"\n🧪 TESTING {detector_name}")
    print("-" * 40)
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train: {len(X_train)} samples ({y_train.sum()} attacks)")
        print(f"Test: {len(X_test)} samples ({y_test.sum()} attacks)")
        
        start_time = time.time()
        detector = detector_class(**kwargs)
        
        # Handle different training approaches (unsupervised vs supervised)
        if "baseline" in detector_name.lower() or "unsupervised" in detector_name.lower():
            # Unsupervised - train only on normal data
            X_train_normal = X_train[y_train == 0]
            detector.fit(X_train_normal)
        else:
            # Supervised
            detector.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = detector.predict(X_test)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"✅ Results:")
        print(f"   F1: {f1:.3f} ({f1*100:.1f}%)")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   Time: {train_time:.1f}s")
        print(f"   Attacks found: {cm[1,1]}/{y_test.sum()}")
        print(f"   False positives: {cm[0,1]}")
        
        return {
            'method': detector_name,
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_time': train_time,
            'attacks_detected': int(cm[1,1]),
            'total_attacks': int(y_test.sum()),
            'false_positives': int(cm[0,1]),
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {'method': detector_name, 'status': 'FAILED', 'error': str(e)}