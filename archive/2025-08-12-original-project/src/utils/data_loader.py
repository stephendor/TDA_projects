import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_ctdapd_clean():
    """Load CTDAPD dataset WITHOUT data leakage features"""
    print("Loading CTDAPD dataset (CLEAN VERSION - NO LEAKAGE)...")
    
    data_path = "data/apt_datasets/Cybersecurity Threat and Awareness Program/CTDAPD Dataset.csv"
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Attack distribution:")
    print(df['Label'].value_counts())
    
    # Create comprehensive attack categories
    def categorize_attack(row):
        if row['Label'] == 'Normal':
            return 'Normal'
        elif row['Attack_Vector'] == 'DDoS':
            return 'DDoS'
        elif row['Attack_Vector'] == 'Brute Force':
            return 'Brute_Force'
        elif row['Attack_Vector'] == 'SQL Injection':
            return 'SQL_Injection'
        else:
            return 'Other_Attack'
    
    df['Attack_Category'] = df.apply(categorize_attack, axis=1)
    print(f"\nFinal attack categories:")
    print(df['Attack_Category'].value_counts())
    
    # CLEAN feature selection - REMOVING ALL POTENTIAL LEAKAGE
    # Removed: IDS_Alert_Count, Anomaly_Score, Anomaly_Severity_Index (detection outputs)
    # Removed: Attack_Vector, Attack_Severity (target-related info)
    clean_feature_cols = [
        'Flow_Duration', 'Packet_Size', 'Flow_Bytes_per_s', 'Flow_Packets_per_s',
        'Total_Forward_Packets', 'Total_Backward_Packets', 
        'Packet_Length_Mean_Forward', 'Packet_Length_Mean_Backward',
        'IAT_Forward', 'IAT_Backward', 'Active_Duration', 'Idle_Duration',
        'CPU_Utilization', 'Memory_Utilization', 'Normalized_Packet_Flow'
    ]
    
    print(f"Clean features (NO LEAKAGE): {len(clean_feature_cols)} features")
    print(f"Features: {clean_feature_cols}")
    
    # Handle missing values and infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    X = df[clean_feature_cols].values
    y = df['Attack_Category'].values
    
    return X, y, df, clean_feature_cols

def load_real_cicids_infiltration():
    """Load the REAL CIC-IDS2017 infiltration dataset - NO SYNTHETIC FALLBACK"""
    print("🎯 LOADING REAL CIC-IDS2017 INFILTRATION DATA")
    print("=" * 60)
    
    # REAL data path - no alternatives
    real_data_path = "data/apt_datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    
    print(f"Loading: {real_data_path}")
    
    # Load the full real dataset
    df = pd.read_csv(real_data_path)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    
    print(f"✅ Dataset shape: {df.shape}")
    print(f"✅ Label distribution:")
    print(df['Label'].value_counts())
    
    # Prepare features (all numeric columns except Label)
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    # Handle missing/infinite values properly
    print(f"🔧 Cleaning data...")
    print(f"   Missing values: {X.isnull().sum().sum()}")
    print(f"   Infinite values: {np.isinf(X.values).sum()}")
    
    # Fill missing with median
    X = X.fillna(X.median())
    
    # Replace infinite values with column medians
    for col in X.columns:
        col_median = X[col].median()
        X[col] = X[col].replace([np.inf, -np.inf], col_median)
    
    # Create binary labels (BENIGN=0, Infiltration=1)
    y = (df['Label'] == 'Infiltration').astype(int)
    
    print(f"✅ Final features: {X.shape[1]} dimensions")
    print(f"✅ Final attack rate: {y.mean():.4%} ({y.sum()} attacks)")
    print(f"✅ Data ready for TDA validation")
    
    return X.values, y.values

def load_sampled_real_data(max_benign=5000):
    """Load real CIC-IDS2017 data but sample benign for speed"""
    print("🎯 LOADING SAMPLED REAL CIC-IDS2017 DATA")
    print("-" * 50)
    
    real_data_path = "data/apt_datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    
    df = pd.read_csv(real_data_path)
    df.columns = df.columns.str.strip()
    
    print(f"Original dataset: {df.shape}")
    
    # Separate attacks and benign
    attacks = df[df['Label'] == 'Infiltration']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"Original attacks: {len(attacks)}")
    print(f"Original benign: {len(benign)}")
    
    # Keep ALL attacks, sample benign for speed
    benign_sampled = benign.sample(n=min(max_benign, len(benign)), random_state=42)
    
    # Combine
    df_sampled = pd.concat([attacks, benign_sampled])
    
    print(f"Sampled dataset: {df_sampled.shape}")
    print(f"Attack rate: {len(attacks)}/{len(df_sampled)} = {len(attacks)/len(df_sampled):.3%}")
    
    # Prepare features
    feature_cols = [col for col in df_sampled.columns if col != 'Label']
    X = df_sampled[feature_cols].select_dtypes(include=[np.number])
    
    # Clean data
    X = X.fillna(X.median())
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
    
    # Binary labels
    y = (df_sampled['Label'] == 'Infiltration').astype(int)
    
    print(f"Final features: {X.shape[1]} dimensions")
    print(f"Final samples: {len(X)} ({y.sum()} attacks)")
    
    return X.values, y.values

def load_and_prepare_cicids2017(max_samples=5000, use_synthetic_labels_if_no_label=False):
    """Load and prepare CIC-IDS2017 data with correct path and optional synthetic labels"""
    
    data_path = Path("data/apt_datasets/cicids2017/MachineLearningCSV/MachineLearningCVE")
    
    logger.info(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Find infiltration data (APT-like attacks)
    infiltration_files = list(data_path.glob("*Infilteration*.csv"))
    
    if not infiltration_files:
        logger.warning("No infiltration files found, using any available data")
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found")
        infiltration_files = [csv_files[0]]
    
    logger.info(f"Found {len(infiltration_files)} infiltration files")
    
    # Load data
    dfs = []
    for file in infiltration_files[:2]:  # Limit to 2 files for faster testing
        logger.info(f"Loading: {file.name}")
        df = pd.read_csv(file, nrows=max_samples//len(infiltration_files))
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    logger.info(f"Loaded {len(data)} total samples")
    
    # Handle common column naming variations
    label_columns = [col for col in data.columns if 'label' in col.lower()]
    if not label_columns:
        if use_synthetic_labels_if_no_label:
            logger.warning("No label column found, creating synthetic labels")
            data['Label'] = 'BENIGN'
            # Mark potential anomalies based on high values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Use z-score to identify outliers
                z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
                anomaly_mask = (z_scores > 3).any(axis=1)
                data.loc[anomaly_mask, 'Label'] = 'ATTACK'
        else:
            raise ValueError("No label column found and synthetic labels not enabled.")
    
    # Get features (all numeric columns except label)
    label_col = [col for col in data.columns if 'label' in col.lower()][0]
    feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in feature_cols:
        feature_cols.remove(label_col)
    
    # Clean data
    X = data[feature_cols].fillna(0)
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
    
    y = data[label_col]
    
    # Convert labels to binary
    y_binary = (y != 'BENIGN').astype(int)
    
    # Take sample to avoid memory issues
    if len(X) > max_samples:
        sample_idx = np.random.choice(len(X), max_samples, replace=False)
        X = X.iloc[sample_idx]
        y_binary = y_binary.iloc[sample_idx]
    
    logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Attack samples: {y_binary.sum()}/{len(y_binary)} ({y_binary.mean():.2%})")
    
    return X.values, y_binary.values