print("="*60)
print("ABLATION STUDY: AE vs EIF vs HYBRID (V13)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
import joblib
import os
import warnings
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from math import sqrt

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# 1. KONFIGURASI
# =============================================================================
DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
EPOCHS = 10
BATCH_SIZE = 2048         
AE_PERCENTILE = 85
EIF_NTREES = 400          
EIF_SAMPLE_SIZE = 256     
EIF_DEPTH = 20   

# =============================================================================
# [STEP A] DATA PREPARATION (SAMA UNTUK SEMUA MODEL)
# =============================================================================
print("\n[PREP] Preparing Data & Features...")

df_banksim = pd.read_csv(DATASET_PATH)

def clean_quotes(value):
    if isinstance(value, str):
        clean_val = value.replace('"', '').replace("'", '').strip()
        try:
            return float(clean_val) if '.' in clean_val else int(clean_val)
        except ValueError:
            return clean_val
    return value

for col in df_banksim.columns:
    df_banksim[col] = df_banksim[col].apply(clean_quotes)

df_banksim = df_banksim.rename(columns={'zipcodeOri': 'customer_zip', 'zipMerchant': 'merchant_zip', 'fraud': 'fraud'})
df_banksim['amount'] = pd.to_numeric(df_banksim['amount'], errors='coerce')
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

df_feat = df_banksim.copy()

# Feature Engineering (Sama seperti V13)
df_feat['hour_of_day'] = df_feat['step'] % 24
df_feat['day_of_week'] = (df_feat['step'] // 24) % 7
df_feat['R_NightTime'] = df_feat['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
df_feat['R_Weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

cust_stats = df_feat.groupby('customer')['amount'].agg(['mean', 'std']).reset_index()
cust_stats.columns = ['customer', 'c_mean', 'c_std']
df_feat = pd.merge(df_feat, cust_stats, on='customer', how='left')
df_feat['c_std'] = df_feat['c_std'].fillna(1).replace(0, 1)
df_feat['User_ZScore'] = (df_feat['amount'] - df_feat['c_mean']) / df_feat['c_std']
df_feat['R_User_Outlier'] = (df_feat['User_ZScore'] > 3).astype(int)

cat_stats = df_feat.groupby('category')['amount'].mean().reset_index(name='cat_mean')
df_feat = pd.merge(df_feat, cat_stats, on='category', how='left')
df_feat['R_Cat_Spike'] = (df_feat['amount'] > (df_feat['cat_mean'] * 2)).astype(int)

step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='tx_per_step')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
df_feat['R_RapidFire'] = (df_feat['tx_per_step'] > 1).astype(int)

df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

df_feat['age'] = df_feat['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

scaler = RobustScaler()
df_feat['Amount_Scaled'] = scaler.fit_transform(df_feat[['amount']])
df_feat['ZScore_Scaled'] = scaler.fit_transform(df_feat[['User_ZScore']])

rule_cols = ['R_NightTime', 'R_Weekend', 'R_User_Outlier', 'R_Cat_Spike', 'R_RapidFire', 'R_DiffZip']
numeric_cols = ['Amount_Scaled', 'ZScore_Scaled']

X_omega = pd.concat([df_feat[rule_cols], df_feat[numeric_cols], df_encoded], axis=1)
y = df_feat['fraud']

# Splitting
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"Data Ready. Test Size: {len(y_test)}")

# =============================================================================
# MODEL 1: AE ONLY (BASELINE)
# =============================================================================
print("\n[EXP 1] Running AutoEncoder Only...")
input_dim = X_train_ae.shape[1]
hidden_dim = max(14, int(input_dim / 1.5)) 
bottleneck_dim = 7

input_layer = Input(shape=(input_dim,))
encoder1 = Dense(hidden_dim, activation='tanh')(input_layer)
bottleneck = Dense(bottleneck_dim, activation='relu')(encoder1)
decoder1 = Dense(hidden_dim, activation='tanh')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoder1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=0,
                callbacks=[EarlyStopping(patience=15)])

# Thresholding AE
def get_mse(data):
    rec = autoencoder.predict(data, verbose=0)
    return np.mean(np.power(data - rec, 2), axis=1)

mse_train = get_mse(X_train_ae.iloc[:10000])
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
mse_test = get_mse(X_test)

# Prediksi AE Only
pred_ae = (mse_test > THRESHOLD).astype(int)

# =============================================================================
# DATA PREP FOR EIF VARIANTS
# =============================================================================
def get_hybrid_features(data):
    if hasattr(data, 'values'): data = data.values
    enc = encoder_model.predict(data, verbose=0)
    rec = autoencoder.predict(data, verbose=0)
    mse = np.mean(np.power(data - rec, 2), axis=1).reshape(-1, 1)
    # Cosine Sim
    norm_d = np.linalg.norm(data, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    cos = (np.sum(data * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    return np.hstack([enc, mse, cos])

# Data Hybrid
print("[PREP] Generating Hybrid Features...")
hyb_train = np.hstack([X_train_eif_raw.values, get_hybrid_features(X_train_eif_raw)])
hyb_test = np.hstack([X_test.values, get_hybrid_features(X_test)])

# Data Raw (Untuk EIF Only)
raw_train = X_train_eif_raw.values
raw_test = X_test.values

# =============================================================================
# FUNGSI EVALUASI MODULAR
# =============================================================================
results = []
def evaluate_model(name, y_true, y_pred, elapsed):
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'Time(s)': elapsed})
    print(f"   -> {name}: F1={f1:.4f} | Recall={rec:.4f}")

# Simpan hasil AE Only
evaluate_model("AE Only", y_test, pred_ae, 0)

# =============================================================================
# MODEL 2: EIF ONLY (RAW DATA, NO HYBRID)
# =============================================================================
print("\n[EXP 2] Running EIF (Standard) on Raw Data...")
start = time.time()
# plus=False artinya Isolation Forest biasa (Standard)
eif_raw = ExtendedIsolationForest(plus=False, n_estimators=EIF_NTREES, max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
eif_raw.fit(raw_train)
scores_raw = eif_raw.predict(raw_test)
pred_eif_raw = (scores_raw > 0.5).astype(int) 
evaluate_model("EIF (Raw)", y_test, pred_eif_raw, time.time()-start)

# =============================================================================
# MODEL 3: EIF+ ONLY (RAW DATA, PLUS EXTENSION)
# =============================================================================
print("\n[EXP 3] Running EIF+ (Extended) on Raw Data...")
start = time.time()
# plus=True artinya Extended Isolation Forest (Hyperplane split)
eif_plus_raw = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
eif_plus_raw.fit(raw_train)
scores_plus_raw = eif_plus_raw.predict(raw_test)
pred_eif_plus_raw = (scores_plus_raw > 0.5).astype(int)
evaluate_model("EIF+ (Raw)", y_test, pred_eif_plus_raw, time.time()-start)

# =============================================================================
# MODEL 4: AE-EIF (HYBRID, STANDARD IF)
# =============================================================================
print("\n[EXP 4] Running AE-EIF (Hybrid, Standard IF)...")
start = time.time()
# Gatekeeper AE
suspects_idx = np.where(pred_ae == 1)[0]
pred_ae_eif = np.zeros(len(y_test))

if len(suspects_idx) > 0:
    # Train Standard IF on Hybrid Data
    eif_hyb_std = ExtendedIsolationForest(plus=False, n_estimators=EIF_NTREES, max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
    eif_hyb_std.fit(hyb_train)
    
    # Predict only suspects
    scores_suspects = eif_hyb_std.predict(hyb_test[suspects_idx])
    pred_ae_eif[suspects_idx] = (scores_suspects > 0.5).astype(int)

evaluate_model("AE-EIF (Hybrid)", y_test, pred_ae_eif, time.time()-start)

# =============================================================================
# MODEL 5: AE-EIF+ (PROPOSED MODEL)
# =============================================================================
print("\n[EXP 5] Running AE-EIF+ (Proposed Hybrid Extended)...")
start = time.time()
# Gatekeeper AE
suspects_idx = np.where(pred_ae == 1)[0]
pred_ae_eif_plus = np.zeros(len(y_test))

if len(suspects_idx) > 0:
    # Train Extended IF on Hybrid Data
    eif_hyb_plus = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
    eif_hyb_plus.fit(hyb_train)
    
    # Predict only suspects
    scores_suspects_plus = eif_hyb_plus.predict(hyb_test[suspects_idx])
    pred_ae_eif_plus[suspects_idx] = (scores_suspects_plus > 0.5).astype(int)

evaluate_model("AE-EIF+ (Proposed)", y_test, pred_ae_eif_plus, time.time()-start)

# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "="*60)
print("FINAL ABLATION STUDY RESULTS")
print("="*60)
df_res = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print(df_res.to_string(index=False))

print("\n[ANALYSIS GUIDE]")
print("1. AE Only: High Recall usually, but Low Precision (High False Positives).")
print("2. EIF (Raw): Good baseline, but struggles with high-dim noise.")
print("3. EIF+ (Raw): Better than EIF, handles complex boundaries better.")
print("4. AE-EIF: Improvements from dimensionality reduction (AE).")
print("5. AE-EIF+: Combines Latent Features (AE) + Complex Splits (Plus). Target: Best F1.")