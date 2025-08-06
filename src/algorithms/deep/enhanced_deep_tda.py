#!/usr/bin/env python3
"""
Enhanced Deep TDA with Kill Chain Awareness
Multi-attack detection with privilege escalation, reconnaissance, lateral movement, and exfiltration patterns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import gudhi as gd
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import base components
from .deep_tda_breakthrough import DifferentiablePersistentHomology, PersistentAttentionLayer

class KillChainAwareEncoder(nn.Module):
    """
    Neural network that understands different phases of APT kill chain
    """
    
    def __init__(self, input_dim=80, hidden_dim=256, kill_chain_phases=6):
        super(KillChainAwareEncoder, self).__init__()
        
        self.kill_chain_phases = kill_chain_phases
        
        # Phase-specific encoders
        self.reconnaissance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        self.privilege_escalation_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        self.lateral_movement_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2), 
            nn.Dropout(0.2)
        )
        
        self.exfiltration_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        
        # Phase attention mechanism
        self.phase_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
    def forward(self, x):
        # Extract phase-specific features
        recon_features = self.reconnaissance_encoder(x)
        priv_features = self.privilege_escalation_encoder(x)
        lateral_features = self.lateral_movement_encoder(x)
        exfil_features = self.exfiltration_encoder(x)
        
        # Combine all phase features
        combined_features = torch.cat([
            recon_features, priv_features, 
            lateral_features, exfil_features
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(combined_features)
        
        # Apply phase attention (add sequence dimension for attention)
        fused_features_seq = fused_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        attended_features, attention_weights = self.phase_attention(
            fused_features_seq, fused_features_seq, fused_features_seq
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        return attended_features, attention_weights

class MultiAttackTDATransformer(nn.Module):
    """
    Enhanced TDA Transformer for multi-attack detection
    """
    
    def __init__(self, input_dim=80, embed_dim=256, num_layers=6, num_heads=8, num_classes=2):
        super(MultiAttackTDATransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Kill chain aware encoder
        self.kill_chain_encoder = KillChainAwareEncoder(input_dim, embed_dim)
        
        # Topological feature extraction
        self.temporal_tda = DifferentiablePersistentHomology(max_dimension=2, num_landmarks=50)
        self.graph_tda = DifferentiablePersistentHomology(max_dimension=1, num_landmarks=30)
        
        # Persistent attention layers (expecting 6 persistence features each)
        self.temporal_attention = PersistentAttentionLayer(embed_dim, num_heads=num_heads)
        self.graph_attention = PersistentAttentionLayer(embed_dim, num_heads=num_heads)
        
        # Multi-scale transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Multi-attack classification heads
        self.attack_type_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 5)  # DDoS, PortScan, WebAttacks, Infiltration, Benign
        )
        
        self.binary_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Kill chain phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 6)  # 6 kill chain phases
        )
        
    def extract_topological_features(self, x):
        """Extract multi-scale topological features"""
        batch_size = x.shape[0]
        
        # Create 2D point clouds for TDA (similar to the base implementation)
        temporal_point_clouds = []
        graph_point_clouds = []
        
        for i in range(batch_size):
            sample = x[i]
            
            # Create temporal embedding for TDA (2D point cloud)
            seq_len = min(20, x.shape[1])  # Limit sequence length
            
            # Temporal coordinates (x-axis: time, y-axis: feature intensity)
            temporal_coords = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            feature_intensity = torch.mean(sample[:seq_len], dim=-1) if len(sample.shape) > 1 else sample[:seq_len]
            
            # Stack into 2D point cloud
            temp_points = torch.stack([temporal_coords, feature_intensity], dim=1)  # (seq_len, 2)
            temporal_point_clouds.append(temp_points.unsqueeze(0))  # Add batch dimension
            
            # Create graph-like point cloud (distance-based embedding)
            if seq_len > 3:
                # Use pairwise feature distances as 2D coordinates
                sample_truncated = sample[:seq_len]
                if len(sample_truncated.shape) == 1:
                    sample_truncated = sample_truncated.unsqueeze(-1)
                
                # Simple 2D projection: mean and std of features
                x_coords = torch.mean(sample_truncated, dim=-1)
                y_coords = torch.std(sample_truncated, dim=-1) if sample_truncated.shape[-1] > 1 else x_coords * 0.1
                
                graph_points = torch.stack([x_coords, y_coords], dim=1)  # (seq_len, 2)
                graph_point_clouds.append(graph_points.unsqueeze(0))  # Add batch dimension
            else:
                # Fallback for small sequences
                graph_points = temp_points
                graph_point_clouds.append(graph_points.unsqueeze(0))
        
        # Stack all point clouds
        temporal_batch = torch.cat(temporal_point_clouds, dim=0)  # (batch_size, seq_len, 2)
        graph_batch = torch.cat(graph_point_clouds, dim=0)  # (batch_size, seq_len, 2)
        
        # Extract topological features using the neural networks
        try:
            temporal_features = self.temporal_tda(temporal_batch)  # Should return (batch_size, 6)
            graph_features = self.graph_tda(graph_batch)  # Should return (batch_size, 6)
            
            # Ensure features are the right shape
            if temporal_features.shape[-1] != 6:
                temporal_features = temporal_features[:, :6] if temporal_features.shape[-1] > 6 else F.pad(temporal_features, (0, 6 - temporal_features.shape[-1]))
            if graph_features.shape[-1] != 6:
                graph_features = graph_features[:, :6] if graph_features.shape[-1] > 6 else F.pad(graph_features, (0, 6 - graph_features.shape[-1]))
                
        except Exception as e:
            print(f"   Warning: TDA computation failed ({str(e)}), using zero features")
            temporal_features = torch.zeros((batch_size, 6), device=x.device)
            graph_features = torch.zeros((batch_size, 6), device=x.device)
        
        return temporal_features, graph_features
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Kill chain aware encoding
        kill_chain_features, attention_weights = self.kill_chain_encoder(x)
        
        # Extract topological features
        temporal_tda, graph_tda = self.extract_topological_features(x)  # Each: (batch_size, 6)
        
        # Apply persistent attention to kill chain features using TDA features
        # Create sequence dimension for attention (using kill chain features as sequence)
        kill_chain_seq = kill_chain_features.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Apply persistent attention using TDA features as attention guides
        temporal_attended = self.temporal_attention(kill_chain_seq, temporal_tda).squeeze(1)
        graph_attended = self.graph_attention(kill_chain_seq, graph_tda).squeeze(1)
        
        # Combine all features (all should be embed_dim size now)
        combined_features = kill_chain_features + temporal_attended + graph_attended  # Element-wise addition
        
        # Ensure correct embedding dimension
        if combined_features.shape[1] != self.embed_dim:
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(combined_features.shape[1], self.embed_dim).to(x.device)
            combined_features = self.feature_projection(combined_features)
        
        # Add sequence dimension for transformer
        sequence_features = combined_features.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Apply transformer encoding
        transformer_output = self.transformer_encoder(sequence_features)
        
        # Remove sequence dimension
        final_features = transformer_output.squeeze(1)
        
        # Multi-head classification
        binary_logits = self.binary_classifier(final_features)
        attack_type_logits = self.attack_type_classifier(final_features)
        phase_logits = self.phase_classifier(final_features)
        
        return {
            'binary_logits': binary_logits,
            'attack_type_logits': attack_type_logits, 
            'phase_logits': phase_logits,
            'features': final_features,
            'attention_weights': attention_weights
        }

class EnhancedDeepTDAAnalyzer:
    """
    Enhanced Deep TDA Analyzer with Kill Chain Awareness and Multi-Attack Detection
    """
    
    def __init__(self, input_dim=80, embed_dim=256, num_layers=6, num_heads=8):
        self.model = MultiAttackTDATransformer(
            input_dim=input_dim,
            embed_dim=embed_dim, 
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Multi-attack type mapping
        self.attack_types = {
            0: 'BENIGN',
            1: 'DDoS', 
            2: 'PortScan',
            3: 'WebAttacks',
            4: 'Infiltration'
        }
        
        # Kill chain phases
        self.kill_chain_phases = {
            0: 'Reconnaissance',
            1: 'Initial Access',
            2: 'Execution', 
            3: 'Privilege Escalation',
            4: 'Lateral Movement',
            5: 'Exfiltration'
        }
        
        print("🚀 ENHANCED DEEP TDA ANALYZER INITIALIZED")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Attack types: {len(self.attack_types)}")
        print(f"   Kill chain phases: {len(self.kill_chain_phases)}")
    
    def prepare_multi_attack_data(self, attack_datasets, max_samples_per_attack=2000):
        """
        Prepare data from multiple attack types for training
        """
        print(f"\\n📊 PREPARING MULTI-ATTACK TRAINING DATA")
        print("-" * 60)
        
        all_data = []
        all_binary_labels = []
        all_attack_type_labels = []
        all_phase_labels = []
        
        attack_type_map = {
            'ddos': 1,
            'portscan': 2, 
            'webattacks': 3,
            'infiltration': 4
        }
        
        # Phase mapping based on attack type
        phase_map = {
            'ddos': 2,  # Execution
            'portscan': 0,  # Reconnaissance
            'webattacks': 1,  # Initial Access
            'infiltration': 4,  # Lateral Movement
        }
        
        for attack_type, df in attack_datasets.items():
            print(f"   Processing {attack_type.upper()}: {len(df):,} samples")
            
            # Extract features
            feature_columns = [
                'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count',
                'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                'Average Packet Size', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
                'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean'
            ]
            
            available_features = [col for col in feature_columns if col in df.columns]
            if len(available_features) < 10:
                print(f"     ⚠️ Limited features available: {len(available_features)}")
                continue
                
            X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
            
            # Balance attack and benign samples
            attack_mask = df['Label'] != 'BENIGN'
            attacks = X[attack_mask]
            benign = X[~attack_mask]
            
            # Sample appropriately
            n_attack_samples = min(len(attacks), max_samples_per_attack // 2)
            n_benign_samples = min(len(benign), max_samples_per_attack // 2)
            
            if n_attack_samples > 0:
                attack_sample = attacks.sample(n=n_attack_samples, random_state=42)
                attack_binary_labels = [1] * n_attack_samples
                attack_type_labels = [attack_type_map.get(attack_type, 0)] * n_attack_samples
                attack_phase_labels = [phase_map.get(attack_type, 0)] * n_attack_samples
                
                all_data.append(attack_sample)
                all_binary_labels.extend(attack_binary_labels)
                all_attack_type_labels.extend(attack_type_labels)
                all_phase_labels.extend(attack_phase_labels)
                
                print(f"     ✅ Attack samples: {n_attack_samples}")
            
            if n_benign_samples > 0:
                benign_sample = benign.sample(n=n_benign_samples, random_state=42)
                benign_binary_labels = [0] * n_benign_samples
                benign_type_labels = [0] * n_benign_samples  # BENIGN
                benign_phase_labels = [0] * n_benign_samples  # No specific phase
                
                all_data.append(benign_sample)
                all_binary_labels.extend(benign_binary_labels)
                all_attack_type_labels.extend(benign_type_labels)
                all_phase_labels.extend(benign_phase_labels)
                
                print(f"     ✅ Benign samples: {n_benign_samples}")
        
        if not all_data:
            raise ValueError("No data could be prepared for training")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Ensure consistent feature dimensions
        target_features = 80  # Target number of features
        if combined_data.shape[1] > target_features:
            combined_data = combined_data.iloc[:, :target_features]
        elif combined_data.shape[1] < target_features:
            # Pad with zeros
            padding_cols = target_features - combined_data.shape[1]
            padding_df = pd.DataFrame(0, index=combined_data.index, 
                                    columns=[f'pad_{i}' for i in range(padding_cols)])
            combined_data = pd.concat([combined_data, padding_df], axis=1)
        
        print(f"\\n✅ MULTI-ATTACK DATA PREPARED")
        print(f"   Total samples: {len(combined_data):,}")
        print(f"   Features: {combined_data.shape[1]}")
        print(f"   Attack rate: {np.mean(all_binary_labels)*100:.1f}%")
        print(f"   Attack types distribution:")
        
        for attack_id, attack_name in self.attack_types.items():
            count = all_attack_type_labels.count(attack_id)
            print(f"     {attack_name}: {count} ({count/len(all_attack_type_labels)*100:.1f}%)")
        
        return (combined_data.values, 
                np.array(all_binary_labels),
                np.array(all_attack_type_labels), 
                np.array(all_phase_labels))
    
    def train(self, X, y_binary, y_attack_type, y_phase, epochs=50, batch_size=64, learning_rate=1e-4):
        """
        Train the enhanced model on multi-attack data
        """
        print(f"\\n🎯 TRAINING ENHANCED DEEP TDA MODEL")
        print("-" * 60)
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_bin_train, y_bin_test, y_type_train, y_type_test, y_phase_train, y_phase_test = train_test_split(
            X_scaled, y_binary, y_attack_type, y_phase, 
            test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_bin_train),
            torch.LongTensor(y_type_train), 
            torch.LongTensor(y_phase_train)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_bin_test),
            torch.LongTensor(y_type_test),
            torch.LongTensor(y_phase_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        binary_criterion = nn.CrossEntropyLoss()
        attack_type_criterion = nn.CrossEntropyLoss()
        phase_criterion = nn.CrossEntropyLoss()
        
        print(f"\\n   Training samples: {len(train_dataset):,}")
        print(f"   Test samples: {len(test_dataset):,}")
        print(f"   Starting training...")
        
        # Training loop
        best_f1 = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_idx, (data, y_bin, y_type, y_phase) in enumerate(train_loader):
                data = data.to(self.device)
                y_bin = y_bin.to(self.device)
                y_type = y_type.to(self.device)
                y_phase = y_phase.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Multi-task losses
                binary_loss = binary_criterion(outputs['binary_logits'], y_bin)
                attack_type_loss = attack_type_criterion(outputs['attack_type_logits'], y_type)
                phase_loss = phase_criterion(outputs['phase_logits'], y_phase)
                
                # Combined loss with weights
                total_loss = binary_loss + 0.5 * attack_type_loss + 0.3 * phase_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation phase
            if epoch % 5 == 0:
                val_f1 = self._evaluate(test_loader)
                avg_train_loss = np.mean(train_losses)
                
                print(f"   Epoch {epoch:2d}: Loss={avg_train_loss:.4f}, Val F1={val_f1:.3f}")
                
                scheduler.step(avg_train_loss)
                
                # Save best model
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\\n✅ TRAINING COMPLETE")
        print(f"   Best validation F1-score: {best_f1:.3f}")
        
        return best_f1
    
    def _evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, y_bin, _, _ in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                preds = torch.argmax(outputs['binary_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_bin.numpy())
        
        f1 = f1_score(all_targets, all_preds, average='weighted')
        return f1
    
    def evaluate_comprehensive(self, X, y_binary, y_attack_type, y_phase):
        """
        Comprehensive evaluation with multi-task performance
        """
        print(f"\\n📊 COMPREHENSIVE EVALUATION")
        print("-" * 60)
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Binary predictions
            binary_preds = torch.argmax(outputs['binary_logits'], dim=1).cpu().numpy()
            
            # Attack type predictions  
            attack_type_preds = torch.argmax(outputs['attack_type_logits'], dim=1).cpu().numpy()
            
            # Phase predictions
            phase_preds = torch.argmax(outputs['phase_logits'], dim=1).cpu().numpy()
        
        # Calculate metrics
        binary_f1 = f1_score(y_binary, binary_preds, average='weighted')
        attack_type_f1 = f1_score(y_attack_type, attack_type_preds, average='weighted')
        phase_f1 = f1_score(y_phase, phase_preds, average='weighted')
        
        print(f"   Binary Classification F1: {binary_f1:.3f}")
        print(f"   Attack Type F1: {attack_type_f1:.3f}")
        print(f"   Kill Chain Phase F1: {phase_f1:.3f}")
        
        # Detailed binary classification report
        print(f"\\n📈 BINARY CLASSIFICATION REPORT:")
        binary_report = classification_report(y_binary, binary_preds, 
                                            target_names=['Benign', 'Attack'])
        print(binary_report)
        
        return {
            'binary_f1': binary_f1,
            'attack_type_f1': attack_type_f1,
            'phase_f1': phase_f1,
            'binary_predictions': binary_preds,
            'attack_type_predictions': attack_type_preds,
            'phase_predictions': phase_preds
        }


def main():
    """Example usage of Enhanced Deep TDA"""
    
    print("🚀 ENHANCED DEEP TDA WITH KILL CHAIN AWARENESS")
    print("=" * 70)
    print("Multi-Attack Detection + APT Kill Chain Phase Recognition")
    print("=" * 70)
    
    # This would be called with real multi-attack datasets
    # For now, just demonstrate initialization
    analyzer = EnhancedDeepTDAAnalyzer(
        input_dim=80,
        embed_dim=256,
        num_layers=6,
        num_heads=8
    )
    
    print(f"\\n✅ Enhanced Deep TDA Analyzer ready for multi-attack training")
    print(f"   Target performance: >85% F1-score")
    print(f"   Kill chain phases: {list(analyzer.kill_chain_phases.values())}")
    print(f"   Attack types: {list(analyzer.attack_types.values())}")


if __name__ == "__main__":
    main()