import numpy as np
import pandas as pd
import os
import pickle
import gc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config_colab import (
    DATA_PATH,
    OUTPUT_PATH,
    SDN_FEATURES,
    TARGET,
    DTYPE_DICT,
    TARGET_CLASSES,
)


def load_csv_optimized(filepath, features, target):
    """Load a single CSV with optimized memory usage"""
    usecols = [f for f in features if f != target] + [target]
    usecols = [c for c in usecols if c in pd.read_csv(filepath, nrows=0).columns]

    dtype_subset = {k: v for k, v in DTYPE_DICT.items() if k in usecols}

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype_subset, low_memory=True)
    return df


def load_all_data_optimized(data_path=DATA_PATH, sample_ratio=1.0):
    """Load all CSV files with memory optimization"""
    all_dfs = []
    total_rows = 0

    for i in range(1, 13):
        filepath = os.path.join(data_path, f"home{i}_labeled.csv")
        if os.path.exists(filepath):
            print(f"Loading {filepath}...")

            df = load_csv_optimized(filepath, SDN_FEATURES, TARGET)

            if sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)

            all_dfs.append(df)
            total_rows += len(df)
            print(
                f"  Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
            )

            gc.collect()

    if all_dfs:
        print(f"\nConcatenating {len(all_dfs)} dataframes...")
        df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()
        print(f"Total shape: {df.shape}")
        return df

    return None


def preprocess_data_efficient(
    df, features=SDN_FEATURES, target=TARGET, target_classes=TARGET_CLASSES
):
    """Preprocess data with memory efficiency"""
    print("Preprocessing data...")

    valid_features = [f for f in features if f in df.columns]

    df = df.dropna(subset=[target]).copy()

    df = df[df[target].isin(target_classes)]
    print(f"  Filtered to {len(target_classes)} target classes: {len(df)} samples")

    for col in valid_features:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0).astype("float32")

    df = df.replace([np.inf, -np.inf], 0)

    X = df[valid_features].values.astype("float32")
    y = df[target].values

    del df
    gc.collect()

    return X, y, valid_features


def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    with open(os.path.join(OUTPUT_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")

    return y_encoded, label_encoder, num_classes


def scale_features_incremental(X, batch_size=100000):
    """Scale features in batches for memory efficiency"""
    print("Scaling features...")

    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    n_samples = X.shape[0]

    for i in range(0, n_samples, batch_size):
        batch = X[i : i + batch_size]
        minmax_scaler.partial_fit(batch)

    for i in range(0, n_samples, batch_size):
        batch = X[i : i + batch_size]
        X[i : i + batch_size] = minmax_scaler.transform(batch)

    for i in range(0, n_samples, batch_size):
        batch = X[i : i + batch_size]
        standard_scaler.partial_fit(batch)

    for i in range(0, n_samples, batch_size):
        batch = X[i : i + batch_size]
        X[i : i + batch_size] = standard_scaler.transform(batch)

    with open(os.path.join(OUTPUT_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump({"minmax": minmax_scaler, "standard": standard_scaler}, f)

    print(f"Scaled feature matrix shape: {X.shape}")
    return X


def prepare_data_colab(sample_ratio=1.0):
    """Main function to prepare data for Colab"""
    print("=" * 60)
    print("Loading and Preprocessing Data (Colab Optimized)")
    print("=" * 60)

    df = load_all_data_optimized(sample_ratio=sample_ratio)

    X, y, feature_names = preprocess_data_efficient(df)
    del df
    gc.collect()

    print(f"\nFeature matrix shape: {X.shape}")

    y_encoded, label_encoder, num_classes = encode_labels(y)
    del y
    gc.collect()

    X_scaled = scale_features_incremental(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, label_encoder, num_classes, feature_names
