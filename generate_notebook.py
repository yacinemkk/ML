#!/usr/bin/env python3
"""Generate the corrected IoT Colab Training notebook."""
import json

def code_cell(cell_id, source_lines):
    return {
        "cell_type": "code",
        "metadata": {"id": cell_id},
        "execution_count": None,
        "outputs": [],
        "source": source_lines,
    }

def md_cell(cell_id, source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": source_lines,
    }

# Read the pipeline source to extract function code
with open("ipfix_home_pipeline.py", "r") as f:
    pipeline_src = f.read()

cells = []

# ── Header ──
cells.append(md_cell("header", [
    "# IoT Device Identification with Adversarial Robustness\n",
    "## Google Colab Pipeline (IoT IPFIX Home)\n",
    "\n",
    "### Includes:\n",
    "1. Full preprocessing & training pipeline (Steps 1-6)\n",
    "2. **STEP 7** – Load saved models from Drive & CORRECTED final evaluation\n",
    "3. Proper MIXED evaluation: clean + adversarial data together\n",
    "4. Detector routes: adversarial → Model-2 (Robust), clean → Model-3 (Base)\n",
]))

# ── Setup ──
cells.append(code_cell("setup", [
    "!pip install xgboost scikit-learn pandas numpy matplotlib seaborn tensorflow -q\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
]))

# ── Imports ──
cells.append(code_cell("imports", [
    "import os, warnings, pickle, gc\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (f1_score, precision_score, recall_score,\n",
    "                             classification_report, accuracy_score, confusion_matrix)\n",
    "import xgboost as xgb\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential, load_model\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "np.random.seed(42)\n",
    "tf.random.set_seed(42)\n",
]))

# ── Config ──
# Extract lines 45-117 from pipeline
config_lines = pipeline_src.split('\n')[44:117]
config_src = [l + '\n' for l in config_lines]
cells.append(code_cell("config", config_src))

# ── Step 1: Data Loading ──
step1_lines = pipeline_src.split('\n')[119:196]
step1_src = [l + '\n' for l in step1_lines]
cells.append(code_cell("step1", step1_src))

# ── Step 2: Base Classifiers ──
step2_lines = pipeline_src.split('\n')[197:260]
step2_src = [l + '\n' for l in step2_lines]
cells.append(code_cell("step2", step2_src))

# ── Step 3: Adversarial Attack Generation ──
step3_lines = pipeline_src.split('\n')[262:446]
step3_src = [l + '\n' for l in step3_lines]
cells.append(code_cell("step3", step3_src))

# ── Step 4: Detectors ──
step4_lines = pipeline_src.split('\n')[449:531]
step4_src = [l + '\n' for l in step4_lines]
cells.append(code_cell("step4", step4_src))

# ── Step 5: Robust ──
step5_lines = pipeline_src.split('\n')[533:636]
step5_src = [l + '\n' for l in step5_lines]
cells.append(code_cell("step5", step5_src))

# ── Step 6: Two-Tiered (original) ──
step6_lines = pipeline_src.split('\n')[639:740]
step6_src = [l + '\n' for l in step6_lines]
cells.append(code_cell("step6", step6_src))

# ── Utilities (_predict, evaluate) ──
util_lines = pipeline_src.split('\n')[743:794]
util_src = [l + '\n' for l in util_lines]
cells.append(code_cell("utilities", util_src))

# ── Plotting utilities ──
plot_lines = pipeline_src.split('\n')[797:928]
plot_src = [l + '\n' for l in plot_lines]
cells.append(code_cell("plotting", plot_src))

# ── Main function ──
main_lines = pipeline_src.split('\n')[930:996]
main_src = [l + '\n' for l in main_lines]
cells.append(code_cell("main", main_src))

# ── Run main ──
cells.append(code_cell("run_main", [
    "# Uncomment to run full training pipeline\n",
    "# main()\n",
]))

# ════════════════════════════════════════════════════
# STEP 7 – LOAD MODELS FROM DRIVE & CORRECTED FINAL TEST
# ════════════════════════════════════════════════════

cells.append(md_cell("step7_header", [
    "---\n",
    "# STEP 7 – Load Saved Models & Corrected Final Evaluation\n",
    "\n",
    "## Logique corrigée du Two-Tiered Defense (Figure 4 du paper) :\n",
    "1. On prépare un jeu de test **MIXTE** : données clean + données adversariales\n",
    "2. Le **Détecteur (Model-1)** prédit pour chaque échantillon : clean (0) ou adversarial (1)\n",
    "3. Si **détecté adversarial** → envoyé au **Model-2 (Robust classifier)**\n",
    "4. Si **détecté clean** → envoyé au **Model-3 (Base classifier)**\n",
    "5. Métriques calculées sur le jeu **MIXTE complet**\n",
    "\n",
    "### Corrections par rapport au code précédent :\n",
    "- ❌ **Avant** : clean et adversarial évalués séparément → métriques biaisées\n",
    "- ✅ **Maintenant** : jeu MIXTE (comme dans le paper), recovery = F1_defended / F1_clean\n",
    "- ❌ **Avant** : test sur seulement `X_test[:n_adv]` (sous-ensemble)\n",
    "- ✅ **Maintenant** : test sur tout le jeu de test\n",
]))

# ── Load models from Drive ──
cells.append(code_cell("load_models", [
    "# ═══════════════════════════════════════════════════\n",
    "# STEP 7A – Load all saved models from Google Drive\n",
    "# ═══════════════════════════════════════════════════\n",
    "\n",
    "MODEL_PATH = '/content/drive/MyDrive/results_ml_avc/'\n",
    "\n",
    "def load_sklearn_model(filepath):\n",
    "    with open(filepath, 'rb') as f:\n",
    "        return pickle.load(f)\n",
    "\n",
    "# ── Load Model-3 (Base classifiers) ──\n",
    "print('Loading Base Classifiers (Model-3)...')\n",
    "base_models = {}\n",
    "for name, fname in [('RF', 'model3_rf.pkl'), ('KNN', 'model3_knn.pkl'),\n",
    "                     ('XGBoost', 'model3_xgboost.pkl')]:\n",
    "    fpath = os.path.join(MODEL_PATH, fname)\n",
    "    if os.path.exists(fpath):\n",
    "        base_models[name] = load_sklearn_model(fpath)\n",
    "        print(f'  ✓ {name} loaded from {fname}')\n",
    "    else:\n",
    "        print(f'  ✗ {name} NOT FOUND: {fpath}')\n",
    "\n",
    "dnn_path = os.path.join(MODEL_PATH, 'model3_dnn.h5')\n",
    "if os.path.exists(dnn_path):\n",
    "    base_models['DNN'] = load_model(dnn_path)\n",
    "    print(f'  ✓ DNN loaded from model3_dnn.h5')\n",
    "else:\n",
    "    print(f'  ✗ DNN NOT FOUND: {dnn_path}')\n",
    "\n",
    "# ── Load Model-1 (Detectors) ──\n",
    "print('\\nLoading Detectors (Model-1)...')\n",
    "detector_models = {}\n",
    "for name, fname in [('RF', 'model1_rf_detector.pkl'), ('KNN', 'model1_knn_detector.pkl'),\n",
    "                     ('XGBoost', 'model1_xgboost_detector.pkl')]:\n",
    "    fpath = os.path.join(MODEL_PATH, fname)\n",
    "    if os.path.exists(fpath):\n",
    "        detector_models[name] = load_sklearn_model(fpath)\n",
    "        print(f'  ✓ {name} loaded from {fname}')\n",
    "    else:\n",
    "        print(f'  ✗ {name} NOT FOUND: {fpath}')\n",
    "\n",
    "det_dnn_path = os.path.join(MODEL_PATH, 'model1_dnn_detector.h5')\n",
    "if os.path.exists(det_dnn_path):\n",
    "    detector_models['DNN'] = load_model(det_dnn_path)\n",
    "    print(f'  ✓ DNN loaded from model1_dnn_detector.h5')\n",
    "else:\n",
    "    print(f'  ✗ DNN NOT FOUND: {det_dnn_path}')\n",
    "\n",
    "# ── Load Model-2 (Robust classifiers) ──\n",
    "print('\\nLoading Robust Classifiers (Model-2)...')\n",
    "robust_models = {}\n",
    "for name, fname in [('RF', 'model2_rf_robust.pkl'), ('KNN', 'model2_knn_robust.pkl'),\n",
    "                     ('XGBoost', 'model2_xgboost_robust.pkl')]:\n",
    "    fpath = os.path.join(MODEL_PATH, fname)\n",
    "    if os.path.exists(fpath):\n",
    "        robust_models[name] = load_sklearn_model(fpath)\n",
    "        print(f'  ✓ {name} loaded from {fname}')\n",
    "    else:\n",
    "        print(f'  ✗ {name} NOT FOUND: {fpath}')\n",
    "\n",
    "rob_dnn_path = os.path.join(MODEL_PATH, 'model2_dnn_robust.h5')\n",
    "if os.path.exists(rob_dnn_path):\n",
    "    robust_models['DNN'] = load_model(rob_dnn_path)\n",
    "    print(f'  ✓ DNN loaded from model2_dnn_robust.h5')\n",
    "else:\n",
    "    print(f'  ✗ DNN NOT FOUND: {rob_dnn_path}')\n",
    "\n",
    "# ── Load label encoder & scaler ──\n",
    "print('\\nLoading Label Encoder & Scaler...')\n",
    "with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'rb') as f:\n",
    "    label_encoder = pickle.load(f)\n",
    "with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb') as f:\n",
    "    scalers = pickle.load(f)\n",
    "\n",
    "label_classes = list(label_encoder.classes_)\n",
    "num_classes = len(label_classes)\n",
    "print(f'  Classes ({num_classes}): {label_classes}')\n",
    "print(f'\\n✅ All models loaded successfully!')\n",
]))

# ── Prepare test data ──
cells.append(code_cell("prepare_test_data", [
    "# ═══════════════════════════════════════════════════\n",
    "# STEP 7B – Prepare test data (clean + adversarial)\n",
    "# ═══════════════════════════════════════════════════\n",
    "\n",
    "# Load and preprocess the full dataset\n",
    "df = load_all_data(DATA_PATH, sample_ratio=SAMPLE_RATIO)\n",
    "X, y, feature_names = preprocess(df)\n",
    "del df; gc.collect()\n",
    "\n",
    "# Encode and scale\n",
    "# Re-use the SAVED scaler/encoder to ensure consistency with trained models\n",
    "y_enc = label_encoder.transform(y)\n",
    "X_mm = scalers['minmax'].transform(X)\n",
    "X_sc = scalers['standard'].transform(X_mm)\n",
    "del X; gc.collect()\n",
    "\n",
    "# Same 75/25 split as training (same random_state=42)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_sc, y_enc, test_size=0.25, random_state=42, stratify=y_enc\n",
    ")\n",
    "n_features = X_train.shape[1]\n",
    "print(f'Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}')\n",
    "print(f'Features: {n_features}')\n",
]))

# ── Generate adversarial test samples ──
cells.append(code_cell("gen_adv_test", [
    "# ═══════════════════════════════════════════════════\n",
    "# STEP 7C – Generate adversarial samples from TEST set\n",
    "# ═══════════════════════════════════════════════════\n",
    "\n",
    "# Generate adversarial samples from a portion of the test set\n",
    "N_ADV_TEST = min(5000, len(X_test) // 4)\n",
    "\n",
    "atk = AdversarialAttackGenerator(feature_names)\n",
    "atk.fit_centroids(X_train, y_train)\n",
    "\n",
    "# Take a random subset of test samples for adversarial generation\n",
    "np.random.seed(42)\n",
    "adv_indices = np.random.choice(len(X_test), N_ADV_TEST, replace=False)\n",
    "X_test_for_adv = X_test[adv_indices]\n",
    "y_test_for_adv = y_test[adv_indices]\n",
    "\n",
    "print(f'Generating {N_ADV_TEST} adversarial samples from test set...')\n",
    "X_adv_test = atk.generate_batch(X_test_for_adv, y_test_for_adv)\n",
    "print(f'✅ Generated {len(X_adv_test)} adversarial test samples')\n",
]))

# ── CORRECTED Two-Tiered Defense with MIXED data ──
cells.append(md_cell("mixed_eval_header", [
    "## STEP 7D – CORRECTED Two-Tiered Defense Evaluation\n",
    "\n",
    "**Key difference from the old code:**\n",
    "- We create a **MIXED** test set: all clean test samples + all adversarial test samples\n",
    "- The detector decides for EACH sample whether it's adversarial or not\n",
    "- Routed accordingly: adversarial → robust model, clean → base model\n",
    "- Metrics computed on the **full mixed dataset**\n",
]))

cells.append(code_cell("corrected_eval", [
    "# ═══════════════════════════════════════════════════\n",
    "# STEP 7D – CORRECTED Two-Tiered Defense (Figure 4)\n",
    "# ═══════════════════════════════════════════════════\n",
    "#\n",
    "# Pipeline (Figure 4 du paper):\n",
    "#   Incoming flow → Model-1 (Detector)\n",
    "#     → Flagged adversarial → Model-2 (Robust classifier)\n",
    "#     → Flagged clean       → Model-3 (Base classifier)\n",
    "#\n",
    "# CORRECTION: Evaluate on MIXED data (clean + adversarial together)\n",
    "# This matches the paper's evaluation methodology.\n",
    "\n",
    "print('=' * 70)\n",
    "print('CORRECTED Two-Tiered Defense Evaluation (MIXED clean + adversarial)')\n",
    "print('=' * 70)\n",
    "\n",
    "# ── Build the MIXED test set ──\n",
    "# Clean: full test set with true device labels\n",
    "# Adversarial: adversarial versions of some test samples with their true device labels\n",
    "X_mixed = np.vstack([X_test, X_adv_test])\n",
    "y_mixed_true = np.hstack([y_test, y_test_for_adv])  # true device labels for all\n",
    "\n",
    "# Ground truth for detector: 0=clean, 1=adversarial\n",
    "y_is_adv = np.hstack([np.zeros(len(X_test)), np.ones(len(X_adv_test))])\n",
    "\n",
    "print(f'Mixed test set: {len(X_mixed)} samples')\n",
    "print(f'  Clean:       {len(X_test)} samples')\n",
    "print(f'  Adversarial: {len(X_adv_test)} samples')\n",
    "\n",
    "all_results = []\n",
    "\n",
    "for det_name, detector in detector_models.items():\n",
    "    print(f'\\n{\"─\" * 60}')\n",
    "    print(f'Detector: {det_name}')\n",
    "    print(f'{\"─\" * 60}')\n",
    "\n",
    "    # ── Step 1: Detector predicts on the MIXED set ──\n",
    "    if det_name == 'DNN':\n",
    "        det_pred = (detector.predict(X_mixed) > 0.5).astype(int).flatten()\n",
    "    else:\n",
    "        det_pred = detector.predict(X_mixed)\n",
    "\n",
    "    # Detector performance\n",
    "    det_acc = accuracy_score(y_is_adv, det_pred)\n",
    "    det_pr = precision_score(y_is_adv, det_pred, zero_division=0)\n",
    "    det_rc = recall_score(y_is_adv, det_pred, zero_division=0)\n",
    "    det_f1 = f1_score(y_is_adv, det_pred, zero_division=0)\n",
    "    print(f'  Detector perf: Acc={det_acc:.4f} Pr={det_pr:.4f} Rc={det_rc:.4f} F1={det_f1:.4f}')\n",
    "\n",
    "    n_flagged_adv = det_pred.sum()\n",
    "    n_flagged_clean = len(det_pred) - n_flagged_adv\n",
    "    print(f'  Routing: {int(n_flagged_clean)} → Base (Model-3), {int(n_flagged_adv)} → Robust (Model-2)')\n",
    "\n",
    "    for cls_name in base_models:\n",
    "        if cls_name not in robust_models:\n",
    "            continue\n",
    "        base_model = base_models[cls_name]\n",
    "        robust_model = robust_models[cls_name]\n",
    "\n",
    "        # ── Step 2: Route through pipeline ──\n",
    "        # For samples detected as clean (0) → use base model (Model-3)\n",
    "        # For samples detected as adversarial (1) → use robust model (Model-2)\n",
    "        pred_base = _predict(base_model, X_mixed)\n",
    "        pred_robust = _predict(robust_model, X_mixed)\n",
    "        y_pred_defended = np.where(det_pred == 0, pred_base, pred_robust)\n",
    "\n",
    "        # ── Metrics WITH defense (on mixed set) ──\n",
    "        acc_def = accuracy_score(y_mixed_true, y_pred_defended)\n",
    "        pr_def = precision_score(y_mixed_true, y_pred_defended, average='weighted', zero_division=0)\n",
    "        rc_def = recall_score(y_mixed_true, y_pred_defended, average='weighted', zero_division=0)\n",
    "        f1_def = f1_score(y_mixed_true, y_pred_defended, average='weighted', zero_division=0)\n",
    "\n",
    "        # ── Metrics WITHOUT defense (base model on mixed set) ──\n",
    "        acc_nodef = accuracy_score(y_mixed_true, pred_base)\n",
    "        pr_nodef = precision_score(y_mixed_true, pred_base, average='weighted', zero_division=0)\n",
    "        rc_nodef = recall_score(y_mixed_true, pred_base, average='weighted', zero_division=0)\n",
    "        f1_nodef = f1_score(y_mixed_true, pred_base, average='weighted', zero_division=0)\n",
    "\n",
    "        # ── Metrics on CLEAN only (base model on clean test) ──\n",
    "        pred_base_clean = _predict(base_model, X_test)\n",
    "        f1_clean = f1_score(y_test, pred_base_clean, average='weighted', zero_division=0)\n",
    "        pr_clean = precision_score(y_test, pred_base_clean, average='weighted', zero_division=0)\n",
    "        rc_clean = recall_score(y_test, pred_base_clean, average='weighted', zero_division=0)\n",
    "        acc_clean = accuracy_score(y_test, pred_base_clean)\n",
    "\n",
    "        # ── Metrics WITHOUT defense on ADV ONLY ──\n",
    "        pred_base_adv = _predict(base_model, X_adv_test)\n",
    "        f1_adv_nodef = f1_score(y_test_for_adv, pred_base_adv, average='weighted', zero_division=0)\n",
    "        pr_adv_nodef = precision_score(y_test_for_adv, pred_base_adv, average='weighted', zero_division=0)\n",
    "        rc_adv_nodef = recall_score(y_test_for_adv, pred_base_adv, average='weighted', zero_division=0)\n",
    "\n",
    "        # ── Metrics WITH defense on ADV ONLY ──\n",
    "        if det_name == 'DNN':\n",
    "            det_adv_only = (detector.predict(X_adv_test) > 0.5).astype(int).flatten()\n",
    "        else:\n",
    "            det_adv_only = detector.predict(X_adv_test)\n",
    "        pred_base_adv2 = _predict(base_model, X_adv_test)\n",
    "        pred_robust_adv = _predict(robust_model, X_adv_test)\n",
    "        y_pred_adv_def = np.where(det_adv_only == 0, pred_base_adv2, pred_robust_adv)\n",
    "        f1_adv_def = f1_score(y_test_for_adv, y_pred_adv_def, average='weighted', zero_division=0)\n",
    "        pr_adv_def = precision_score(y_test_for_adv, y_pred_adv_def, average='weighted', zero_division=0)\n",
    "        rc_adv_def = recall_score(y_test_for_adv, y_pred_adv_def, average='weighted', zero_division=0)\n",
    "\n",
    "        # Recovery = F1_defended / F1_clean (how much of clean performance is recovered)\n",
    "        recovery = (f1_adv_def / f1_clean * 100) if f1_clean > 0 else 0\n",
    "\n",
    "        print(f'\\n  Classifier: {cls_name}')\n",
    "        print(f'    Clean (no attack)        | Pr={pr_clean:.4f} Rc={rc_clean:.4f} F1={f1_clean:.4f}')\n",
    "        print(f'    Adv WITHOUT defense      | Pr={pr_adv_nodef:.4f} Rc={rc_adv_nodef:.4f} F1={f1_adv_nodef:.4f}')\n",
    "        print(f'    Adv WITH defense         | Pr={pr_adv_def:.4f} Rc={rc_adv_def:.4f} F1={f1_adv_def:.4f}')\n",
    "        print(f'    Mixed WITH defense       | Pr={pr_def:.4f} Rc={rc_def:.4f} F1={f1_def:.4f}')\n",
    "        print(f'    Recovery (F1_adv_def/F1_clean) = {recovery:.1f}%')\n",
    "\n",
    "        all_results.append({\n",
    "            'detector': det_name, 'classifier': cls_name,\n",
    "            'precision_clean': pr_clean, 'recall_clean': rc_clean, 'f1_clean': f1_clean,\n",
    "            'precision_adv_nodef': pr_adv_nodef, 'recall_adv_nodef': rc_adv_nodef, 'f1_adv_nodef': f1_adv_nodef,\n",
    "            'precision_adv_defended': pr_adv_def, 'recall_adv_defended': rc_adv_def, 'f1_adv_defended': f1_adv_def,\n",
    "            'precision_mixed_defended': pr_def, 'recall_mixed_defended': rc_def, 'f1_mixed_defended': f1_def,\n",
    "            'recovery_pct': recovery,\n",
    "        })\n",
    "\n",
    "df_results = pd.DataFrame(all_results)\n",
    "df_results.to_csv(os.path.join(OUTPUT_PATH, 'step7_corrected_defense_results.csv'), index=False)\n",
    "print(f'\\n✅ Results saved to {OUTPUT_PATH}step7_corrected_defense_results.csv')\n",
    "df_results\n",
]))

# ── Plot Figure 10 corrected ──
cells.append(code_cell("plot_fig10_corrected", [
    "# ═══════════════════════════════════════════════════\n",
    "# CORRECTED Figure 10: With vs Without Defense\n",
    "# ═══════════════════════════════════════════════════\n",
    "# Use the BEST detector (highest F1) for plotting\n",
    "\n",
    "# Find best detector\n",
    "det_f1s = {}\n",
    "for det_name in detector_models:\n",
    "    if det_name == 'DNN':\n",
    "        dp = (detector_models[det_name].predict(X_mixed) > 0.5).astype(int).flatten()\n",
    "    else:\n",
    "        dp = detector_models[det_name].predict(X_mixed)\n",
    "    det_f1s[det_name] = f1_score(y_is_adv, dp, zero_division=0)\n",
    "best_det = max(det_f1s, key=det_f1s.get)\n",
    "print(f'Best detector: {best_det} (F1={det_f1s[best_det]:.4f})')\n",
    "\n",
    "# Filter results for best detector\n",
    "best_res = df_results[df_results['detector'] == best_det].copy()\n",
    "models_list = best_res['classifier'].tolist()\n",
    "\n",
    "# ── Figure 10 style: grouped bar chart ──\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
    "\n",
    "# Left: Without Defense (Precision, Recall, F1 on adversarial only)\n",
    "ax = axes[0]\n",
    "x = np.arange(len(models_list))\n",
    "width = 0.25\n",
    "colors_nodef = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']\n",
    "\n",
    "bars1 = ax.bar(x - width, best_res['precision_adv_nodef'] * 100, width, label='Precision', color='#4472C4')\n",
    "bars2 = ax.bar(x, best_res['recall_adv_nodef'] * 100, width, label='Recall', color='#ED7D31')\n",
    "bars3 = ax.bar(x + width, best_res['f1_adv_nodef'] * 100, width, label='F1-Score', color='#A5A5A5')\n",
    "\n",
    "for bars in [bars1, bars2, bars3]:\n",
    "    for bar in bars:\n",
    "        h = bar.get_height()\n",
    "        if h > 0:\n",
    "            ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',\n",
    "                    ha='center', va='bottom', fontsize=9, rotation=0)\n",
    "\n",
    "ax.set_ylabel('Score (%)', fontsize=12)\n",
    "ax.set_title('Without Defence', fontsize=14, fontweight='bold')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(models_list, fontsize=11)\n",
    "ax.legend(fontsize=10)\n",
    "ax.set_ylim(0, 110)\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "\n",
    "# Right: With Defense (Precision, Recall, F1 on adversarial with defense)\n",
    "ax = axes[1]\n",
    "bars1 = ax.bar(x - width, best_res['precision_adv_defended'] * 100, width, label='Precision', color='#4472C4')\n",
    "bars2 = ax.bar(x, best_res['recall_adv_defended'] * 100, width, label='Recall', color='#ED7D31')\n",
    "bars3 = ax.bar(x + width, best_res['f1_adv_defended'] * 100, width, label='F1-Score', color='#A5A5A5')\n",
    "\n",
    "for bars in [bars1, bars2, bars3]:\n",
    "    for bar in bars:\n",
    "        h = bar.get_height()\n",
    "        if h > 0:\n",
    "            ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',\n",
    "                    ha='center', va='bottom', fontsize=9, rotation=0)\n",
    "\n",
    "ax.set_ylabel('Score (%)', fontsize=12)\n",
    "ax.set_title('With Defence', fontsize=14, fontweight='bold')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(models_list, fontsize=11)\n",
    "ax.legend(fontsize=10)\n",
    "ax.set_ylim(0, 110)\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "\n",
    "plt.suptitle(f'Figure 10: Performance With Robustness Measures (Detector: {best_det})',\n",
    "             fontsize=15, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(OUTPUT_PATH, 'Figure_10_CORRECTED_Defense.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('✅ Figure 10 (Corrected) saved')\n",
]))

# ── Summary comparison ──
cells.append(code_cell("summary", [
    "# ═══════════════════════════════════════════════════\n",
    "# Summary: Compare all detectors × classifiers\n",
    "# ═══════════════════════════════════════════════════\n",
    "\n",
    "print('\\n' + '=' * 80)\n",
    "print('SUMMARY: Two-Tiered Defense Results (All Detectors × All Classifiers)')\n",
    "print('=' * 80)\n",
    "print(f'{\"Detector\":>10} | {\"Classifier\":>10} | {\"F1 Clean\":>10} | {\"F1 Adv NoD\":>12} | {\"F1 Adv Def\":>12} | {\"Recovery\":>10}')\n",
    "print('-' * 80)\n",
    "for _, row in df_results.iterrows():\n",
    "    print(f'{row[\"detector\"]:>10} | {row[\"classifier\"]:>10} | '\n",
    "          f'{row[\"f1_clean\"]*100:>9.1f}% | '\n",
    "          f'{row[\"f1_adv_nodef\"]*100:>11.1f}% | '\n",
    "          f'{row[\"f1_adv_defended\"]*100:>11.1f}% | '\n",
    "          f'{row[\"recovery_pct\"]:>9.1f}%')\n",
    "print('=' * 80)\n",
]))

# ── Build notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "L4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("IoT_Colab_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✅ Notebook generated: IoT_Colab_Training.ipynb")
print(f"   Total cells: {len(cells)}")
