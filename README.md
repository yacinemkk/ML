# IoT Device Identification with Adversarial Robustness

## Project Overview

This project implements the research paper **"Enhancing SDN-enabled Network Access Control with Adversarial Robust IoT Device Identification"** for detecting IoT devices using machine learning on network flow data, with defense mechanisms against adversarial attacks.

## Project Structure

```
ML/
├── data/                           # Dataset directory
│   ├── home1_labeled.csv
│   ├── home2_labeled.csv
│   ├── ...
│   └── home12_labeled.csv
├── docs/
│   └── Manuscript anonymous.pdf    # Research paper
├── output/                         # Generated outputs
│   ├── *.pkl                       # Saved models
│   ├── *.h5                        # DNN models
│   └── *.csv                       # Results
├── README.md
└── IoT_Device_Identification_Adversarial.ipynb
```

## Implementation Plan

### 1. Data Preprocessing
- **Load**: 12 CSV files (home1 to home12_labeled.csv)
- **Features**: 28 SDN-accessible flow features
- **Normalization**: MinMax [0,1] + Standardization (mean=0, std=1)
- **Split**: 75% train / 25% test

### 2. Device Identification Models

| Model | Hyperparameters |
|-------|-----------------|
| **DNN** | 2 hidden layers, 256 neurons, Adam, ReLU |
| **Random Forest** | n_estimators=100, criterion='gini' |
| **KNN** | n_neighbors=3 |
| **XGBoost** | n_estimators=100, eval_metric='merror' |

### 3. Adversarial Attack Generation
- Iterative evasion algorithm with projection
- Formula: `x_adv = Projection[x₀ + c·t·mask·sign(μ_target - x₀)·|Difference|]`
- L0 minimization (feature selection)
- L2 minimization (perturbation magnitude)
- Semantic constraints

### 4. Defense Mechanisms

#### 4.1 Adversarial Instance Detection (Model-1)
Binary classifier to detect adversarial samples before classification.

| Model | Hyperparameters |
|-------|-----------------|
| **DNN** | 3 layers, 256 neurons, Adam, softmax |
| **RF** | n_estimators=300, max_depth=None |
| **KNN** | n_neighbors=5, algorithm='brute' |
| **XGBoost** | n_estimators=300, lr=0.2, max_depth=3 |

#### 4.2 Adversarial Training (Model-2)
Retraining on mixed benign + adversarial data.

| Model | Hyperparameters |
|-------|-----------------|
| **DNN** | 3 layers, 256 neurons, ReLU |
| **RF** | n_estimators=200, bootstrap=True |
| **KNN** | n_neighbors=3, weights='distance' |
| **XGBoost** | learning_rate=0.2, max_depth=7 |

### 5. Evaluation Metrics
- Precision, Recall, F1-score
- Per-device classification performance
- Impact analysis (before/after attack)
- Defense effectiveness

## Features Used (SDN-Accessible)

```
duration, ipProto, outPacketCount, outByteCount, inPacketCount, inByteCount,
outSmallPktCount, outLargePktCount, outNonEmptyPktCount, outDataByteCount,
outAvgIAT, outFirstNonEmptyPktSize, outMaxPktSize, outStdevPayloadSize,
outStdevIAT, outAvgPacketSize, inSmallPktCount, inLargePktCount,
inNonEmptyPktCount, inDataByteCount, inAvgIAT, inFirstNonEmptyPktSize,
inMaxPktSize, inStdevPayloadSize, inStdevIAT, inAvgPacketSize,
http, https, smb, dns, ntp, tcp, udp, ssdp, lan, wan
```

**Target:** `device` (multi-class classification)

## Output Files

| File | Description |
|------|-------------|
| `rf_model.pkl` | Random Forest model |
| `knn_model.pkl` | KNN model |
| `xgb_model.pkl` | XGBoost model |
| `dnn_model.h5` | DNN model (Keras) |
| `scaler.pkl` | MinMaxScaler |
| `label_encoder.pkl` | LabelEncoder |
| `device_identification_results.csv` | Classification results |
| `adversarial_impact.csv` | Attack impact analysis |
| `defense_results.csv` | Defense effectiveness |
| `per_device_f1_scores.csv` | F1-score per device |

## Dataset Information

- **IoT IPFIX Home**: 12 households, 24 device types, 47 days collection
- **Devices**: Cameras, smart plugs, humidifiers, speakers, bulbs, sensors, etc.
- **Source**: Pashamokhtari et al. (2023)

## Expected Results (from Paper)

### Device Identification (Clean Data)
| Model | F1-Score |
|-------|----------|
| XGBoost | 88.1% |
| RF | 86.9% |
| KNN | 83.0% |
| DNN | 82.1% |

### Adversarial Impact
- F1-score drops by 40-70 percentage points under attack

### Defense Effectiveness
- Adversarial detection: ~90% F1-score
- Recovery with adversarial training: up to 90% of original performance

## Usage

```python
# Load trained models
import pickle
from tensorflow.keras.models import load_model

# Load sklearn models
with open('output/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load DNN model
dnn_model = load_model('output/dnn_model.h5')

# Load preprocessing
with open('output/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

## References

Pashamokhtari, A., et al. (2023). Dynamic inference from IoT traffic flows under concept drifts in residential ISP networks. IEEE Internet of Things Journal.

## Notes

- Data path: `/home/pc/Desktop/ML/data/` or `/content/drive/MyDrive/PFE/IPFIX_ML_Instances`
- Output path: `/home/pc/Desktop/ML/output/`
