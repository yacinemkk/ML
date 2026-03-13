"""
IoT Device Identification with Adversarial Robustness
======================================================
Implementation based on:
  "Enhancing SDN-enabled Network Access Control with Adversarial Robust IoT Device Identification"

Dataset: IoT IPFIX Home (12 households, 18 classes after filtering)

Pipeline:
  Step 1 – Data loading & preprocessing
  Step 2 – Train base classifiers (Model-3) with correct hyperparameters (Table 3 / Table 6)
  Step 3 – Adversarial attack generation (Section 4.2 formulation)
  Step 4 – Adversarial Detector training (Model-1, Table 4)
  Step 5 – Adversarial Training / Robust classifiers (Model-2, Table 5)
  Step 6 – Two-Tiered Defense evaluation (Figure 4)
"""

import os
import warnings
import pickle
import gc

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH   = "/content/drive/MyDrive/PFE/IPFIX_ML_Instances/"
OUTPUT_PATH = "/content/drive/MyDrive/results_ml_avc/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

SAMPLE_RATIO = 1.0   # <1.0 for quick testing

# SDN-compatible features (accessible through SDN controller APIs)
SDN_FEATURES = [
    "duration", "ipProto",
    "outPacketCount", "outByteCount", "inPacketCount", "inByteCount",
    "outSmallPktCount", "outLargePktCount", "outNonEmptyPktCount", "outDataByteCount",
    "outAvgIAT", "outFirstNonEmptyPktSize", "outMaxPktSize",
    "outStdevPayloadSize", "outStdevIAT", "outAvgPacketSize",
    "inSmallPktCount", "inLargePktCount", "inNonEmptyPktCount", "inDataByteCount",
    "inAvgIAT", "inFirstNonEmptyPktSize", "inMaxPktSize",
    "inStdevPayloadSize", "inStdevIAT", "inAvgPacketSize",
    "http", "https", "smb", "dns", "ntp", "tcp", "udp", "ssdp", "lan", "wan",
]

DTYPE_DICT = {
    "duration": "float32", "ipProto": "int16",
    "outPacketCount": "int32", "outByteCount": "int64",
    "inPacketCount": "int32", "inByteCount": "int64",
    "outSmallPktCount": "int32", "outLargePktCount": "int32",
    "outNonEmptyPktCount": "int32", "outDataByteCount": "int64",
    "outAvgIAT": "float32", "outFirstNonEmptyPktSize": "int32",
    "outMaxPktSize": "int32", "outStdevPayloadSize": "float32",
    "outStdevIAT": "float32", "outAvgPacketSize": "float32",
    "inSmallPktCount": "int32", "inLargePktCount": "int32",
    "inNonEmptyPktCount": "int32", "inDataByteCount": "int64",
    "inAvgIAT": "float32", "inFirstNonEmptyPktSize": "int32",
    "inMaxPktSize": "int32", "inStdevPayloadSize": "float32",
    "inStdevIAT": "float32", "inAvgPacketSize": "float32",
    "http": "int8", "https": "int8", "smb": "int8", "dns": "int8",
    "ntp": "int8", "tcp": "int8", "udp": "int8", "ssdp": "int8",
    "lan": "int8", "wan": "int8",
    "device": "category", "name": "category",
}

# 18 classes retained for IoT IPFIX Home (Table 7)
VALID_CLASSES = [
    "eclear", "sleep", "esensor", "hub-plus", "humidifier",
    "home-unit", "inkjet-printer", "smart-wifi-plug-mini", "smart-power-strip",
    "echo-dot", "fire7-tablet", "google-nest-mini", "google-chromecast",
    "atom-cam", "kasa-camera-pro", "kasa-smart-led-lamp", "fire-tv-stick-4k", "qrio-hub",
]

TARGET = "name"

# Feature categories for SDN-constraint-aware perturbation (Section 4.2)
# Independent: can be perturbed freely
# Dependent:   must change consistently with related features
# Non-modifiable: fixed network/protocol identifiers
INDEPENDENT_FEATURES = [
    "outAvgIAT", "outStdevIAT", "inAvgIAT", "inStdevIAT",
    "outAvgPacketSize", "inAvgPacketSize",
    "outFirstNonEmptyPktSize", "inFirstNonEmptyPktSize",
    "outMaxPktSize", "inMaxPktSize",
    "outStdevPayloadSize", "inStdevPayloadSize",
    "duration",
]
DEPENDENT_FEATURES = [
    "outPacketCount", "outByteCount", "inPacketCount", "inByteCount",
    "outSmallPktCount", "outLargePktCount", "outNonEmptyPktCount", "outDataByteCount",
    "inSmallPktCount", "inLargePktCount", "inNonEmptyPktCount", "inDataByteCount",
]
NON_MODIFIABLE_FEATURES = [
    "ipProto", "http", "https", "smb", "dns", "ntp", "tcp", "udp", "ssdp", "lan", "wan",
]


# ─────────────────────────────────────────────
# STEP 1 – DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_csv_optimized(filepath):
    available = pd.read_csv(filepath, nrows=0).columns.tolist()
    usecols = [c for c in SDN_FEATURES + [TARGET] if c in available]
    dtype_subset = {k: v for k, v in DTYPE_DICT.items() if k in usecols}
    return pd.read_csv(filepath, usecols=usecols, dtype=dtype_subset, low_memory=True)


def load_all_data(data_path=DATA_PATH, sample_ratio=1.0):
    print("=" * 60)
    print("STEP 1 – Loading IoT IPFIX Home Dataset")
    print("=" * 60)
    all_dfs = []
    for i in range(1, 13):
        fp = os.path.join(data_path, f"home{i}_labeled.csv")
        if os.path.exists(fp):
            print(f"  Loading home{i}_labeled.csv ...")
            df = load_csv_optimized(fp)
            if sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)
            mem = df.memory_usage(deep=True).sum() / 1024 ** 2
            print(f"    Shape: {df.shape} | Memory: {mem:.1f} MB")
            all_dfs.append(df)
            gc.collect()
    df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    print(f"\nTotal shape: {df.shape}")
    return df


def preprocess(df):
    """Clean, filter classes, handle NaNs/Infs – returns X (float32), y (str)."""
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET].isin(VALID_CLASSES)]
    print(f"  After class filtering: {len(df)} rows, {len(VALID_CLASSES)} classes")

    valid_feats = [f for f in SDN_FEATURES if f in df.columns]

    # Remove duplicate rows (Section 5.1 – step 3)
    before = len(df)
    df = df.drop_duplicates(subset=valid_feats)
    print(f"  Duplicates removed: {before - len(df)}")

    for col in valid_feats:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0).astype("float32")

    df = df.replace([np.inf, -np.inf], 0)

    X = df[valid_feats].values.astype("float32")
    y = df[TARGET].values
    return X, y, valid_feats


def encode_and_scale(X, y):
    """LabelEncode y; MinMax → StandardScale X (Section 5.1 steps 1-2)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"  Classes ({n_classes}): {list(le.classes_)}")

    mm = MinMaxScaler()
    X_mm = mm.fit_transform(X)           # [0, 1] normalisation
    std = StandardScaler()
    X_sc = std.fit_transform(X_mm)       # zero-mean, unit-variance standardisation

    with open(os.path.join(OUTPUT_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(OUTPUT_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump({"minmax": mm, "standard": std}, f)

    return X_sc, y_enc, le, n_classes


# ─────────────────────────────────────────────
# STEP 2 – BASE CLASSIFIERS (Model-3) – Table 3 / Table 6
# ─────────────────────────────────────────────

def build_base_classifiers(num_classes, n_features):
    """Returns dict of base classifiers with paper-exact hyperparameters."""
    dnn = Sequential([
        Dense(256, activation="relu", input_shape=(n_features,)),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    dnn.compile(optimizer=Adam(0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    return {
        "RF": RandomForestClassifier(
            n_estimators=300, criterion="gini", max_depth=None,
            bootstrap=True, max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="distance", algorithm="auto", n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, eval_metric="merror", learning_rate=0.1,
            max_depth=None, random_state=42, n_jobs=-1, tree_method="hist",
        ),
        "DNN": dnn,
    }


def train_base_classifiers(models, X_train, y_train, X_test, y_test, label_classes):
    print("\n" + "=" * 60)
    print("STEP 2 – Training Base Classifiers (Model-3)")
    print("=" * 60)
    trained, results = {}, []
    es = EarlyStopping(patience=3, restore_best_weights=True)

    for name, model in models.items():
        print(f"\n  Training {name} ...")
        if name == "DNN":
            model.fit(X_train, y_train, epochs=30, batch_size=256,
                      validation_split=0.1, callbacks=[es], verbose=0)
            model.save(os.path.join(OUTPUT_PATH, "model3_dnn.h5"))
        else:
            model.fit(X_train, y_train)
            with open(os.path.join(OUTPUT_PATH, f"model3_{name.lower()}.pkl"), "wb") as f:
                pickle.dump(model, f)
        trained[name] = model
        r = evaluate(model, X_test, y_test, name, label_classes)
        results.append(r)

    # Save summary
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_class'} for r in results])
    df_res.to_csv(os.path.join(OUTPUT_PATH, "step2_base_identification_results.csv"), index=False)
    
    # Plot Figure 5
    plot_figure_5(results, label_classes)
    
    return trained


# ─────────────────────────────────────────────
# STEP 3 – ADVERSARIAL ATTACK GENERATION (Section 4.2)
# ─────────────────────────────────────────────

class AdversarialAttackGenerator:
    """
    Implements the paper's iterative centroid-based adversarial perturbation:
      x_adv = Projection[ x0 + c·t·mask·sign(µ_target − x0)·|Difference(µ_target, x0)| ]

    L0 minimisation  → binary masks restrict which features are perturbed
                        (only INDEPENDENT + DEPENDENT, not NON_MODIFIABLE)
    L2 minimisation  → pick from the 3 closest class centroids (K-means)
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.class_centroids = {}   # {class_id: centroid_vector}
        self._build_modifiable_mask(feature_names)

    def _build_modifiable_mask(self, feat_names):
        """Binary mask: 1 = modifiable (independent/dependent), 0 = locked."""
        modifiable = set(INDEPENDENT_FEATURES + DEPENDENT_FEATURES)
        self.mask = np.array(
            [1.0 if f in modifiable else 0.0 for f in feat_names], dtype=np.float32
        )

    def fit_centroids(self, X_train, y_train):
        """Compute per-class mean (centroid) vectors."""
        classes = np.unique(y_train)
        for c in classes:
            self.class_centroids[c] = X_train[y_train == c].mean(axis=0)

    def _three_closest_targets(self, x, true_class):
        """Return the 3 class IDs closest (L2) to x, excluding the true class."""
        dists = {
            c: np.linalg.norm(x - mu)
            for c, mu in self.class_centroids.items()
            if c != true_class
        }
        return sorted(dists, key=dists.get)[:3]

    def _projection(self, x_adv):
        """Clip to valid scaled range and enforce integer rounding for discrete features."""
        return np.clip(x_adv, -3.0, 3.0)

    def generate(self, x, true_class, max_iter=50, c=0.05):
        """
        Generate one adversarial example against the class centroid.
        Tries the 3 closest target classes and returns the best (lowest L2 dist).
        """
        candidates = self._three_closest_targets(x, true_class)
        best_adv, best_dist = x.copy(), np.inf

        for target_class in candidates:
            mu_t = self.class_centroids[target_class]
            x_adv = x.copy()
            for t in range(1, max_iter + 1):
                diff = mu_t - x_adv
                direction = np.sign(diff) * np.abs(diff)
                x_adv = x_adv + c * t * self.mask * direction
                x_adv = self._projection(x_adv)

            dist = np.linalg.norm(x_adv - x)
            if dist < best_dist:
                best_dist = dist
                best_adv = x_adv.copy()

        return best_adv

    def generate_batch(self, X, y, n_samples=None):
        if n_samples is None:
            n_samples = len(X)
        n_samples = min(n_samples, len(X))
        X_adv = []
        for i in range(n_samples):
            x_adv = self.generate(X[i], y[i])
            X_adv.append(x_adv)
            if (i + 1) % 500 == 0:
                print(f"    Adversarial samples generated: {i + 1}/{n_samples}")
        return np.array(X_adv, dtype=np.float32)


def generate_adversarial_samples(base_models, X_train, y_train, X_test, y_test,
                                 feature_names, label_classes, n_adv=5000):
    print("\n" + "=" * 60)
    print("STEP 3 – Adversarial Attack Generation (Section 4.2)")
    print("=" * 60)

    # Fit centroid generator on training data
    atk = AdversarialAttackGenerator(feature_names)
    atk.fit_centroids(X_train, y_train)

    X_sample = X_test[:n_adv]
    y_sample = y_test[:n_adv]

    # Generate adversarial test samples (used to evaluate all 4 models)
    print(f"\n  Generating {n_adv} adversarial samples ...")
    X_adv = atk.generate_batch(X_sample, y_sample)

    # Measure adversarial impact on all 4 base models
    print("\n  Adversarial Impact on Base Models:")
    impact_results = []
    
    # For per-class plotting
    per_class_f1_clean = {}
    per_class_f1_adv = {}
    
    for name, model in base_models.items():
        # Clean evals
        y_pred_c = _predict(model, X_sample)
        f1_clean = f1_score(y_sample, y_pred_c, average="weighted", zero_division=0)
        
        # Adv evals
        y_pred_a = _predict(model, X_adv)
        f1_adv = f1_score(y_sample, y_pred_a, average="weighted", zero_division=0)
        
        drop = f1_clean - f1_adv
        print(f"    {name:10s} | F1_clean={f1_clean:.4f}  F1_adv={f1_adv:.4f}  Drop={drop:.4f}")
        impact_results.append({"model": name, "f1_clean": f1_clean,
                                "f1_adv": f1_adv, "drop": drop})
                                
        # Collect per-class for XGBoost (or best model)
        if name == "XGBoost":
            rep_c = classification_report(y_sample, y_pred_c, output_dict=True, zero_division=0)
            rep_a = classification_report(y_sample, y_pred_a, output_dict=True, zero_division=0)
            for i, cls in enumerate(label_classes):
                idx = str(i)
                per_class_f1_clean[cls] = rep_c[idx]['f1-score'] if idx in rep_c else 0.0
                per_class_f1_adv[cls]   = rep_a[idx]['f1-score'] if idx in rep_a else 0.0

    pd.DataFrame(impact_results).to_csv(
        os.path.join(OUTPUT_PATH, "step3_adversarial_impact.csv"), index=False
    )
    
    # Plot Figure 7
    plot_figure_7(per_class_f1_clean, per_class_f1_adv, label_classes)
    
    return X_adv, X_sample, y_sample, atk


# ─────────────────────────────────────────────
# STEP 4 – ADVERSARIAL DETECTOR – Model-1 (Table 4)
# ─────────────────────────────────────────────

def build_detector_classifiers(n_features):
    """Binary classifiers: benign (0) vs adversarial (1). Table 4 hyperparameters."""
    dnn = Sequential([
        Dense(256, activation="relu", input_shape=(n_features,)),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    dnn.compile(optimizer=Adam(0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"])

    return {
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=None, max_features="sqrt",
            min_samples_split=5, random_state=42, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="uniform", algorithm="brute", n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.2,
            random_state=42, n_jobs=-1, tree_method="hist",
        ),
        "DNN": dnn,
    }


def train_detectors(X_train, y_train_device,  # clean training data
                    X_adv, X_sample_clean,     # adversarial & matched clean test samples
                    n_features):
    print("\n" + "=" * 60)
    print("STEP 4 – Training Adversarial Detectors (Model-1, Table 4)")
    print("=" * 60)

    # Build binary detection dataset: clean=0, adversarial=1
    n_det = min(len(X_adv), len(X_train))
    X_det = np.vstack([X_train[:n_det], X_adv])
    y_det = np.hstack([np.zeros(n_det, dtype=np.int8), np.ones(len(X_adv), dtype=np.int8)])

    X_det_tr, X_det_ts, y_det_tr, y_det_ts = train_test_split(
        X_det, y_det, test_size=0.25, random_state=42, stratify=y_det
    )
    print(f"  Detection dataset – train: {len(X_det_tr)}  test: {len(X_det_ts)}")

    detector_models = build_detector_classifiers(n_features)
    trained_detectors = {}
    det_results = []
    es = EarlyStopping(patience=3, restore_best_weights=True)

    for name, model in detector_models.items():
        print(f"\n  Training Detector {name} ...")
        if name == "DNN":
            model.fit(X_det_tr, y_det_tr, epochs=30, batch_size=256,
                      validation_split=0.1, callbacks=[es], verbose=0)
            y_pred = (model.predict(X_det_ts) > 0.5).astype(int).flatten()
            model.save(os.path.join(OUTPUT_PATH, "model1_dnn_detector.h5"))
        else:
            model.fit(X_det_tr, y_det_tr)
            y_pred = model.predict(X_det_ts)
            with open(os.path.join(OUTPUT_PATH, f"model1_{name.lower()}_detector.pkl"), "wb") as f:
                pickle.dump(model, f)

        f1 = f1_score(y_det_ts, y_pred, average="binary", zero_division=0)
        pr = precision_score(y_det_ts, y_pred, average="binary", zero_division=0)
        rc = recall_score(y_det_ts, y_pred, average="binary", zero_division=0)
        print(f"    {name:10s} | F1={f1:.4f}  Precision={pr:.4f}  Recall={rc:.4f}")
        trained_detectors[name] = model
        det_results.append({"model": name, "f1": f1, "precision": pr, "recall": rc})

    pd.DataFrame(det_results).to_csv(
        os.path.join(OUTPUT_PATH, "step4_detector_results.csv"), index=False
    )
    
    # Plot Figure 9
    plot_figure_9(det_results)
    
    return trained_detectors


# ─────────────────────────────────────────────
# STEP 5 – ADVERSARIAL TRAINING – Model-2 (Table 5)
# ─────────────────────────────────────────────

def build_robust_classifiers(num_classes, n_features):
    """Classifiers retrained on clean+adversarial data. Table 5 hyperparameters."""
    dnn = Sequential([
        Dense(256, activation="relu", input_shape=(n_features,)),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    dnn.compile(optimizer=Adam(0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    return {
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features="sqrt",
            bootstrap=True, random_state=42, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=3, weights="distance", algorithm="auto", n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.2,
            random_state=42, n_jobs=-1, tree_method="hist",
        ),
        "DNN": dnn,
    }


def train_robust_classifiers(base_models, X_train, y_train,
                              X_adv, y_adv_true,
                              X_test, y_test, num_classes, n_features):
    print("\n" + "=" * 60)
    print("STEP 5 – Adversarial Training / Robust Classifiers (Model-2, Table 5)")
    print("=" * 60)

    # Joint corpus: clean training data + adversarial samples with true labels
    X_robust = np.vstack([X_train, X_adv])
    y_robust  = np.hstack([y_train, y_adv_true])
    print(f"  Robust training set size: {len(X_robust)} samples")

    robust_models = build_robust_classifiers(num_classes, n_features)
    trained_robust = {}
    rob_results = []
    es = EarlyStopping(patience=3, restore_best_weights=True)

    # Sample of test set matching adversarial samples (same indices as X_adv)
    n_adv = len(X_adv)
    X_test_sample = X_test[:n_adv]
    y_test_sample = y_test[:n_adv]
    X_adv_test = X_adv  # same adversarial set used for evaluation

    for name, model in robust_models.items():
        print(f"\n  Training Robust {name} ...")
        if name == "DNN":
            model.fit(X_robust, y_robust, epochs=30, batch_size=256,
                      validation_split=0.1, callbacks=[es], verbose=0)
            model.save(os.path.join(OUTPUT_PATH, "model2_dnn_robust.h5"))
        else:
            model.fit(X_robust, y_robust)
            with open(os.path.join(OUTPUT_PATH, f"model2_{name.lower()}_robust.pkl"), "wb") as f:
                pickle.dump(model, f)

        trained_robust[name] = model

        f1_clean = f1_score(y_test_sample, _predict(model, X_test_sample), average="weighted")
        f1_adv   = f1_score(y_test_sample, _predict(model, X_adv_test),    average="weighted")
        f1_base  = f1_score(y_test_sample,
                            _predict(base_models[name], X_adv_test), average="weighted")
        recovery = (f1_adv / f1_clean * 100) if f1_clean > 0 else 0
        print(f"    {name:10s} | F1_clean={f1_clean:.4f}  F1_adv_before={f1_base:.4f}"
              f"  F1_adv_after={f1_adv:.4f}  Recovery={recovery:.1f}%")
        rob_results.append({
            "model": name, "f1_clean": f1_clean,
            "f1_adv_before_robust": f1_base,
            "f1_adv_after_robust": f1_adv,
            "recovery_pct": recovery,
        })

    pd.DataFrame(rob_results).to_csv(
        os.path.join(OUTPUT_PATH, "step5_robust_training_results.csv"), index=False
    )
    return trained_robust


# ─────────────────────────────────────────────
# STEP 6 – TWO-TIERED DEFENSE EVALUATION (Figure 4)
# ─────────────────────────────────────────────

def two_tiered_defense_evaluation(base_models, detector_models, robust_models,
                                   X_test_clean, y_test_clean,
                                   X_adv, y_adv_true):
    """
    Pipeline (Figure 4):
      Incoming flow → Model-1 (Detector)
        → Flagged as adversarial → classify with Model-2 (Robust classifier)
        → Flagged as benign     → classify with Model-3 (Base classifier)
    """
    print("\n" + "=" * 60)
    print("STEP 6 – Two-Tiered Defense Evaluation (Figure 4)")
    print("=" * 60)

    pipeline_results = []

    for det_name, detector in detector_models.items():
        print(f"\n  Detector: {det_name}")

        # ── Evaluate on CLEAN test samples ──
        if det_name == "DNN":
            det_clean = (detector.predict(X_test_clean) > 0.5).astype(int).flatten()
        else:
            det_clean = detector.predict(X_test_clean)

        # ── Evaluate on ADVERSARIAL test samples ──
        if det_name == "DNN":
            det_adv = (detector.predict(X_adv) > 0.5).astype(int).flatten()
        else:
            det_adv = detector.predict(X_adv)

        for cls_name in base_models:
            base_model   = base_models[cls_name]
            robust_model = robust_models[cls_name]

            # ── Clean samples through pipeline ──
            y_pred_clean = np.where(
                det_clean == 0,
                _predict(base_model, X_test_clean),      # benign → Model-3
                _predict(robust_model, X_test_clean),    # flagged → Model-2
            )
            f1_clean = f1_score(y_test_clean, y_pred_clean, average="weighted")

            # ── Adversarial samples through pipeline ──
            y_pred_adv = np.where(
                det_adv == 0,
                _predict(base_model, X_adv),             # not detected → Model-3
                _predict(robust_model, X_adv),           # detected     → Model-2
            )
            f1_adv = f1_score(y_adv_true, y_pred_adv, average="weighted")

            # F1 under attack with NO defense (baseline)
            f1_no_defense = f1_score(y_adv_true,
                                     _predict(base_model, X_adv), average="weighted")
            recovery = (f1_adv / f1_no_defense * 100) if f1_no_defense > 0 else 0

            print(f"    Classifier={cls_name:10s} | "
                  f"F1_clean={f1_clean:.4f}  "
                  f"F1_adv_nodefense={f1_no_defense:.4f}  "
                  f"F1_adv_defended={f1_adv:.4f}  "
                  f"Recovery={recovery:.1f}%")

            pipeline_results.append({
                "detector": det_name, "classifier": cls_name,
                "f1_clean": f1_clean,
                "f1_adv_no_defense": f1_no_defense,
                "f1_adv_defended": f1_adv,
                "recovery_pct": recovery,
            })

    df_res = pd.DataFrame(pipeline_results)
    df_res.to_csv(os.path.join(OUTPUT_PATH, "step6_twotiered_defense_results.csv"), index=False)
    print(f"\n  Results saved to {OUTPUT_PATH}")
    
    # Plot Figure 10 (Using results from the best detector, e.g. DNN)
    plot_figure_10(df_res)
    
    return df_res


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _predict(model, X):
    """Unified predict that handles Keras (softmax) and sklearn models."""
    y_pred = model.predict(X)
    if hasattr(y_pred, "shape") and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return np.argmax(y_pred, axis=1)
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
        return (y_pred > 0.5).astype(int).flatten()
    return y_pred.flatten().astype(int)


def evaluate(model, X_test, y_test, name, label_classes=None):
    y_pred = _predict(model, X_test)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    pr  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rc  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    
    per_class = {}
    if label_classes is not None:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for i, cls in enumerate(label_classes):
            idx = str(i)
            if idx in report:
                per_class[cls] = {"precision": report[idx]["precision"], "recall": report[idx]["recall"], "f1": report[idx]["f1-score"]}
            else:
                per_class[cls] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    print(f"    {name:10s} | F1={f1:.4f}  Precision={pr:.4f}  Recall={rc:.4f}")
    return {"model": name, "f1": f1, "precision": pr, "recall": rc, "per_class": per_class}


# ─────────────────────────────────────────────
# PLOTTING UTILITIES FOR PAPER FIGURES
# ─────────────────────────────────────────────

def plot_figure_5(results, label_classes):
    """Figure 5: IoT Device Identification Scores in IoT IPFIX Home."""
    # We will plot the Precision, Recall, and F1 of the best performing model (e.g., XGBoost)
    xgb_res = next((r for r in results if r["model"] == "XGBoost"), None)
    if not xgb_res or not xgb_res["per_class"]: return
    
    metrics = xgb_res["per_class"]
    df_plot = pd.DataFrame(metrics).T
    
    plt.figure(figsize=(14, 6))
    x = np.arange(len(label_classes))
    width = 0.25
    
    plt.bar(x - width, df_plot["precision"]*100, width, label='Precision')
    plt.bar(x, df_plot["recall"]*100, width, label='Recall')
    plt.bar(x + width, df_plot["f1"]*100, width, label='F1-Score')
    
    plt.xlabel('IoT Devices')
    plt.ylabel('Score (%)')
    plt.title('Figure 5: IoT Device Identification Scores in IoT IPFIX Home (XGBoost)')
    plt.xticks(x, label_classes, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'Figure_5_Device_Identification.png'))
    plt.close()

def plot_figure_7(clean_f1, adv_f1, label_classes):
    """Figure 7: Adversarial effect on IoT IPFIX Home identification device."""
    plt.figure(figsize=(14, 6))
    x = np.arange(len(label_classes))
    width = 0.35
    
    c_f1 = [clean_f1[cls]*100 for cls in label_classes]
    a_f1 = [adv_f1[cls]*100 for cls in label_classes]
    
    plt.bar(x - width/2, c_f1, width, label='Clean F1-Score', color='blue')
    plt.bar(x + width/2, a_f1, width, label='Adversarial F1-Score', color='red')
    
    plt.xlabel('IoT Devices')
    plt.ylabel('F1-Score (%)')
    plt.title('Figure 7: Adversarial effect on IoT IPFIX Home identification device (XGBoost)')
    plt.xticks(x, label_classes, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'Figure_7_Adversarial_Effect.png'))
    plt.close()

def plot_figure_9(det_results):
    """Figure 9: Adversarial Instance Detection."""
    df_plot = pd.DataFrame(det_results)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_plot))
    width = 0.25
    
    plt.bar(x - width, df_plot["precision"]*100, width, label='Precision')
    plt.bar(x, df_plot["recall"]*100, width, label='Recall')
    plt.bar(x + width, df_plot["f1"]*100, width, label='F1-Score')
    
    plt.xlabel('Detection Models')
    plt.ylabel('Score (%)')
    plt.title('Figure 9: Adversarial Instance Detection')
    plt.xticks(x, df_plot["model"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'Figure_9_Adversarial_Detection.png'))
    plt.close()

def plot_figure_10(df_res):
    """Figure 10: Performance Evaluation of Device Identification With Robustness Measures."""
    # Pick the best detector, typically DNN, for the plot
    dnn_res = df_res[df_res['detector'] == 'DNN']
    if dnn_res.empty:
        dnn_res = df_res
        
    models = dnn_res['classifier'].tolist()
    f1_clean = dnn_res['f1_clean'] * 100
    f1_adv_nodef = dnn_res['f1_adv_no_defense'] * 100
    f1_adv_def = dnn_res['f1_adv_defended'] * 100
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, f1_clean, width, label='Clean F1-Score', color='green')
    plt.bar(x, f1_adv_nodef, width, label='F1-Score under Attack', color='red')
    plt.bar(x + width, f1_adv_def, width, label='F1-Score with Robustness', color='blue')
    
    plt.xlabel('Classification Models')
    plt.ylabel('F1-Score (%)')
    plt.title('Figure 10: Evaluation in IoT IPFIX Home With Robustness Measures')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'Figure_10_Robustness_Measures.png'))
    plt.close()


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    # ── Step 1: Load & Preprocess ──────────────────────────────
    df = load_all_data(DATA_PATH, sample_ratio=SAMPLE_RATIO)
    X, y, feature_names = preprocess(df)
    del df; gc.collect()

    X_sc, y_enc, label_encoder, num_classes = encode_and_scale(X, y)
    del X; gc.collect()

    # 75/25 split (Section 5.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    print(f"\n  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    n_features = X_train.shape[1]

    # ── Step 2: Base Classifiers (Model-3) ─────────────────────
    base_models_cfg  = build_base_classifiers(num_classes, n_features)
    base_models      = train_base_classifiers(base_models_cfg, X_train, y_train, X_test, y_test, list(label_encoder.classes_))

    # ── Step 3: Adversarial Attack Generation ──────────────────
    N_ADV = min(5000, len(X_test) // 4)
    X_adv, X_sample, y_sample, attacker = generate_adversarial_samples(
        base_models, X_train, y_train, X_test, y_test, feature_names, list(label_encoder.classes_), n_adv=N_ADV
    )

    # ── Step 4: Train Detectors (Model-1, Table 4) ─────────────
    detector_models = train_detectors(
        X_train, y_train, X_adv, X_sample, n_features
    )

    # ── Step 5: Adversarial Training (Model-2, Table 5) ────────
    robust_models = train_robust_classifiers(
        base_models, X_train, y_train,
        X_adv, y_sample,
        X_test, y_test, num_classes, n_features
    )

    # ── Step 6: Two-Tiered Defense Evaluation (Figure 4) ───────
    two_tiered_defense_evaluation(
        base_models, detector_models, robust_models,
        X_sample, y_sample, X_adv, y_sample
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE – All results saved to:", OUTPUT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
