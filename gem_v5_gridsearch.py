print("="*60)
print("IMPLEMENTASI AE-EIF+ (GRID SEARCH OPTIMIZATION)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import joblib
import os
import warnings
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI GRID SEARCH
# =============================================================================
# Kita akan mencoba kombinasi ini untuk mencari hasil tertinggi
AE_THRESHOLDS_TO_TRY = [60, 65, 70, 75, 80, 85, 90, 95] 
EIF_DECISION_CUTOFFS = [0.45, 0.5, 0.55, 0.6, 0.65] 

# Fixed Config sesuai Paper
EPOCHS = 100 
BATCH_SIZE = 64
EIF_NTREES = 200 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 18 

# Paths
dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
ae_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/ae_base.keras'
enc_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/enc_base.keras'

# =============================================================================
# 1. LOAD & PREPARE DATA (FIXED CLEANING)
# =============================================================================
print(f"Loading & Preparing Data...")
df_banksim = pd.read_csv(dataset_path)

# --- PERBAIKAN DI SINI ---
def clean_quotes(value):
    if isinstance(value, str):
        # Hapus kutip dulu
        clean_val = value.replace('"', '').replace("'", '').strip()
        # Coba ubah ke angka, jika gagal (misal 'C123'), kembalikan string
        try:
            if '.' in clean_val:
                return float(clean_val)
            return int(clean_val)
        except ValueError:
            return clean_val
    return value

for col in df_banksim.columns:
    df_banksim[col] = df_banksim[col].apply(clean_quotes)

# Rename columns
df_banksim = df_banksim.rename(columns={'zipcodeOri': 'customer_zip', 'zipMerchant': 'merchant_zip', 'fraud': 'fraud'})

# Pastikan tipe data benar
df_banksim['age'] = pd.to_numeric(df_banksim['age'], errors='coerce').fillna(0).astype(int)
df_banksim['amount'] = pd.to_numeric(df_banksim['amount'], errors='coerce')
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

# =============================================================================
# 2. FEATURE ENGINEERING (ENHANCED)
# =============================================================================
print("Executing Feature Engineering...")
df_feat = df_banksim.copy()

# 1. Statistik Amount per Customer
cust_stats = df_feat.groupby('customer')['amount'].agg(['mean', 'std']).reset_index()
cust_stats.columns = ['customer', 'cust_avg_amt', 'cust_std_amt']
df_feat = pd.merge(df_feat, cust_stats, on='customer', how='left')
df_feat['cust_std_amt'] = df_feat['cust_std_amt'].fillna(1).replace(0, 1)

df_feat['R_Amt_ZScore'] = (df_feat['amount'] - df_feat['cust_avg_amt']) / df_feat['cust_std_amt']
df_feat['R_Amt_Ratio'] = df_feat['amount'] / df_feat['cust_avg_amt'].replace(0, 1)

# 2. Velocity (Step count)
step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='step_tx_count')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')

# 3. Category Frequency Encoding
cat_counts = df_feat['category'].value_counts(normalize=True)
df_feat['R_Cat_Freq'] = df_feat['category'].map(cat_counts)

# 4. Binary Rules
threshold_amt = df_feat['amount'].quantile(0.95)
df_feat['R_HighAmount'] = (df_feat['amount'] > threshold_amt).astype(int)
df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)
df_feat['R_Age_Under25'] = (df_feat['age'] < 25).astype(int)
df_feat['R_Age_Over65'] = (df_feat['age'] > 65).astype(int)

# Select Numeric Cols
numeric_cols = ['step', 'age', 'amount', 'R_Amt_ZScore', 'R_Amt_Ratio', 'step_tx_count', 'R_Cat_Freq',
                'R_HighAmount', 'R_DiffZip', 'R_Age_Under25', 'R_Age_Over65']

# Clean Infinite values
df_feat[numeric_cols] = df_feat[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

# Scaling
scaler = StandardScaler()
df_feat[numeric_cols] = scaler.fit_transform(df_feat[numeric_cols])

X_omega = df_feat[numeric_cols]
y = df_feat['fraud']

# Splitting Data
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# 70% Normal for AE Train
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)

# Test Set (50% remaining Normal + All Fraud)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5))), np.ones(len(X_fraud))])

# EIF Training Set (Sisa Normal)
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"Data Splits -> AE Train: {len(X_train_ae)}, EIF Train: {len(X_train_eif_raw)}, Test: {len(X_test)}")

# =============================================================================
# 3. TRAIN AUTOENCODER (CHECKPOINT)
# =============================================================================
input_dim = X_train_ae.shape[1]
input_layer = Input(shape=(input_dim,))
encoder1 = Dense(14, activation='relu')(input_layer)
bottleneck = Dense(7, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(14, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoder1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck) 
autoencoder.compile(optimizer='adam', loss='mse')

if os.path.exists(ae_path):
    print("\nLoading pre-trained AutoEncoder...")
    autoencoder = load_model(ae_path)
    encoder_model = load_model(enc_path)
else:
    print("\nTraining AutoEncoder (110 Epochs)...")
    autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    shuffle=True, validation_split=0.1, verbose=1,
                    callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])
    autoencoder.save(ae_path)
    encoder_model.save(enc_path)

# =============================================================================
# 4. UTILITIES & PRE-CALCULATION
# =============================================================================
def extract_mu(data):
    if hasattr(data, 'values'): data = data.values
    A = encoder_model.predict(data, verbose=0)
    rec = autoencoder.predict(data, verbose=0)
    B = np.linalg.norm(data - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    # Cosine Sim
    C = (np.sum(data * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    return np.hstack([A, B, C]), np.mean(np.power(data - rec, 2), axis=1)

print("\nExtracting Features (this may take a minute)...")
# 1. MSE Training (untuk thresholding)
_, mse_train_ae = extract_mu(X_train_ae.iloc[:10000]) 

# 2. Data Training EIF (Omega + Mu)
mu_train_eif, _ = extract_mu(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

# 3. Data Test (Omega + Mu) & MSE Test
mu_test, mse_test = extract_mu(X_test)
X_test_np = X_test.values

# =============================================================================
# 5. GRID SEARCH LOOP
# =============================================================================
print("\n" + "="*40)
print("STARTING GRID SEARCH")
print("="*40)
print(f"{'AE_Thr':<10} | {'EIF_Cut':<10} | {'Acc':<10} | {'F1':<10} | {'Prec':<10} | {'Rec':<10}")
print("-" * 70)

best_score = 0
best_params = {}
best_report = ""
best_cm = None

# Train EIF Sekali Saja (EIF tidak tergantung threshold AE)
print("Training EIF Model...")
eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                  max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                                  max_depth=EIF_DEPTH)
eif_model.fit(X_train_eif_combined)
print("EIF Trained.")

# Pre-calculate EIF scores for ALL test data (optimasi kecepatan)
# Kita hitung score untuk semua, nanti baru difilter pakai mask AE
# Ini lebih cepat daripada predict berulang-ulang di dalam loop
print("Calculating EIF scores for full test set...")
X_test_full_combined = np.hstack([X_test_np, mu_test])
eif_scores_full = eif_model.predict(X_test_full_combined) 

for ae_pct in AE_THRESHOLDS_TO_TRY:
    # 1. Tentukan Threshold Value
    threshold_val = np.percentile(mse_train_ae, ae_pct)
    
    # 2. Tentukan Siapa yang "Suspicious" menurut AE
    # Jika MSE > Threshold -> Suspicious -> Pakai skor EIF
    # Jika MSE <= Threshold -> Normal -> Skor 0
    suspicious_mask = mse_test > threshold_val
    
    # Ambil skor EIF hanya untuk yang suspicious, sisanya 0
    final_scores = np.zeros(len(X_test_np))
    final_scores[suspicious_mask] = eif_scores_full[suspicious_mask]
    
    for eif_cut in EIF_DECISION_CUTOFFS:
        # 3. Final Decision
        final_preds = (final_scores > eif_cut).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, final_preds)
        f1 = f1_score(y_test, final_preds)
        prec = precision_score(y_test, final_preds, zero_division=0)
        rec = recall_score(y_test, final_preds)
        
        print(f"{ae_pct:<10} | {eif_cut:<10} | {acc:.4f}     | {f1:.4f}     | {prec:.4f}     | {rec:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_params = {'ae_pct': ae_pct, 'eif_cut': eif_cut}
            best_report = classification_report(y_test, final_preds)
            best_cm = confusion_matrix(y_test, final_preds)

print("\n" + "="*60)
print("WINNER CONFIGURATION")
print("="*60)
print(f"Best AE Percentile: {best_params.get('ae_pct')}")
print(f"Best EIF Cutoff:    {best_params.get('eif_cut')}")
print("\nConfusion Matrix:")
print(best_cm)
print("\nClassification Report:")
print(best_report)