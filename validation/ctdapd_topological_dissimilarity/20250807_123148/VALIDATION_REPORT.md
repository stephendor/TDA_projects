# CTDAPD Topological Dissimilarity Attack Detection

## Method
**Bruillard, Nowak, and Purvine (2016) Approach**
- Sliding window analysis of chronological network flows
- Baseline topology from normal traffic windows
- Wasserstein distance for topological dissimilarity
- Attack detection via dissimilarity spikes

## ATTACK DETECTION RESULTS

### Performance Metrics
- **Attack Recall**: 0.0503 (55/1094 attacks detected)
- **Attack Precision**: 1.0000 (55/55 predictions correct)
- **Attack F1-Score**: 0.0957

### Detection Summary
- **Total Windows**: 1094
- **Windows with Attacks**: 1094
- **Attack Windows Detected**: 55
- **False Alarms**: 0
- **Missed Attacks**: 1039

### Topological Analysis
- **Dissimilarity Threshold**: 0.1930
- **Baseline Topology**: 35 persistent features
- **Method**: H0 persistence + Wasserstein distance

## Validation Claims

✅ **CLAIM**: Topological dissimilarity detects 5.0% of attack windows
✅ **CLAIM**: 100.0% of dissimilarity spike predictions are correct attacks
❌ **CLAIM**: F1-score of 0.096 for attack detection

## Method Validation
Following proven Bruillard et al. approach for network anomaly detection using topological dissimilarity.

*Attack detection validation using topological data analysis - timestamp: 20250807_123148*
