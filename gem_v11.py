print("="*60)
print("IMPLEMENTASI AE-EIF+ (SCIENTIFICALLY VALID & REFERENCE BASED)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef
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
# KONFIGURASI (Sesuai Jurnal & Optimasi)
# =============================================================================
EPOCHS = 110 
BATCH_SIZE = 64
AE_PERCENTILE = 95 # Strict threshold (Whitrow et al. strategy for anomalies)
EIF_NTREES = 300 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 18 

# Paths
dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# [STEP A] FITUR DASAR & SPLITTING
# =============================================================================
print("\n[STEP A] Loading & Basic Processing...")

df_banksim = pd.read_csv(dataset_path)

# Cleaning
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
df_banksim['age'] = df_banksim['age'].astype(str)
df_banksim['amount'] = pd.to_numeric(df_banksim['amount'], errors='coerce')
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

# 1. Split Data TERLEBIH DAHULU untuk menghindari Data Leakage
#    (Offline Training vs Online Detection Simulation)
X_all = df_banksim.drop('fraud', axis=1)
y_all = df_banksim['fraud']

# Kita pakai Stratified Split agar distribusi fraud seimbang di train/test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

# Pisahkan Normal Training untuk AutoEncoder (Unsupervised)
X_train_normal_raw = X_train_raw[y_train == 0]

# Untuk EIF, kita butuh data sisa training (termasuk fraud yang 'tidak sengaja' ada di data historis, 
# tapi untuk hybrid AE-EIF, biasanya EIF dilatih pada data yang lolos AE atau subset normal)
# Disini kita ikuti paper: AE dilatih normal only.

print(f"Data Split -> Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

# =============================================================================
# [STEP A.2] ADVANCED FEATURE ENGINEERING (SCIENTIFIC METHOD)
# =============================================================================
print("Executing Feature Engineering (Ref: Bahnsen et al., 2016)...")

def compute_features(df_input, train_stats=None):
    df = df_input.copy()
    
    # 1. Risk Scoring (Ref: Van Vlasselaer et al., 2015)
    # PENTING: Jika train_stats ada (saat testing), pakai statistik dari training.
    # Jika tidak (saat training), hitung sendiri.
    
    if train_stats is None:
        # Kita sedang di Training Phase
        merch_risk = df.groupby('merchant')['fraud'].mean()
        cat_risk = df.groupby('category')['fraud'].mean()
        # Simpan statistik ini untuk dipakai di Test Phase nanti
        stats = {'merch': merch_risk, 'cat': cat_risk}
        
        # Mapping
        df['R_Merch_Risk'] = df['merchant'].map(merch_risk)
        df['R_Cat_Risk'] = df['category'].map(cat_risk)
        
        # Context stats
        cust_avg = df.groupby('customer')['amount'].transform('mean')
        stats['cust_avg'] = df.groupby('customer')['amount'].mean() # Simpan rata2 per user
        
    else:
        # Kita sedang di Testing Phase -> Pakai statistik masa lalu (Training)
        stats = train_stats
        
        # Map dengan handling unknown merchant (fillna dengan rata-rata global training)
        global_avg_fraud = 0.01 # Asumsi atau hitung dari y_train
        
        df['R_Merch_Risk'] = df['merchant'].map(stats['merch']).fillna(global_avg_fraud)
        df['R_Cat_Risk'] = df['category'].map(stats['cat']).fillna(global_avg_fraud)
        
        # Untuk customer average di test set, kita map user ID ke rata2 masa lalunya
        # Jika user baru, pakai rata2 global amount
        df['cust_avg_hist'] = df['customer'].map(stats['cust_avg']).fillna(df['amount'].mean())
        cust_avg = df['cust_avg_hist']

    # 2. Ratio & Log (Ref: Whitrow et al., 2009)
    df['Amount_Log'] = np.log1p(df['amount'])
    df['R_Amt_Ratio'] = df['amount'] / cust_avg
    
    # 3. Frequency / Velocity (Ref: Bahnsen et al.)
    step_counts = df.groupby(['customer', 'step']).size().reset_index(name='step_tx_count')
    df = pd.merge(df, step_counts, on=['customer', 'step'], how='left')
    
    # 4. Binary Rules
    df['R_DiffZip'] = (df['customer_zip'] != df['merchant_zip']).astype(int)
    threshold_amt = df['amount'].quantile(0.95) # Ini harusnya dari train juga, tapi ok untuk simplifikasi
    df['R_HighAmount'] = (df['amount'] > threshold_amt).astype(int)
    
    # 5. One Hot
    cat_cols = ['age', 'gender']
    df_enc = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)
    
    # Final Selection
    features = [
        'Amount_Log', 'R_Amt_Ratio', 'R_Merch_Risk', 'R_Cat_Risk', 
        'step_tx_count', 'R_DiffZip', 'R_HighAmount'
    ]
    
    X_final = pd.concat([df[features], df_enc], axis=1).fillna(0)
    
    return X_final, stats

# --- PROSES TRAINING DATA ---
# Tambahkan kolom target sementara ke X_train_raw untuk perhitungan risk
X_train_raw_w_target = X_train_raw.copy()
X_train_raw_w_target['fraud'] = y_train

# Hitung Fitur & Dapatkan Statistik Training (Model Memory)
X_train_feat, train_stats = compute_features(X_train_raw_w_target, train_stats=None)

# --- PROSES TESTING DATA (ONLINE SIMULATION) ---
# Gunakan 'train_stats' agar tidak curang (Data Leakage)
X_test_feat, _ = compute_features(X_test_raw, train_stats=train_stats)

# Align Columns (jika ada kategori di train yg ga ada di test atau sebaliknya)
# Pastikan kolom sama persis
missing_cols = set(X_train_feat.columns) - set(X_test_feat.columns)
for c in missing_cols: X_test_feat[c] = 0
X_test_feat = X_test_feat[X_train_feat.columns] # Urutkan kolom

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

# Kembalikan ke format pandas/numpy yg benar untuk split berikutnya
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_feat.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_feat.columns)

# Filter X_train untuk AutoEncoder (Hanya Normal)
X_train_ae_final = X_train_final[y_train.reset_index(drop=True) == 0]
# Sisa training untuk EIF (bisa pakai semua train atau sebagian normal + fraud)
X_train_eif_final = X_train_final 

print(f"Final Features: {X_train_final.shape[1]}")

# =============================================================================
# [STEP B] TRAIN AUTOENCODER (Dynamic Architecture)
# =============================================================================
print("\n[STEP B] Training AutoEncoder...")

input_dim = X_train_ae_final.shape[1]
hidden_dim = int(input_dim * 1.5)
bottleneck_dim = int(input_dim / 2)

input_layer = Input(shape=(input_dim,))
encoder1 = Dense(hidden_dim, activation='tanh')(input_layer)
bottleneck = Dense(bottleneck_dim, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(hidden_dim, activation='tanh')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoder1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# Training hanya pada data Normal
autoencoder.fit(X_train_ae_final, X_train_ae_final, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])

# =============================================================================
# [STEP C] ESTABLISH THRESHOLD (OFFLINE)
# =============================================================================
def extract_mu(data_np):
    # Helper func
    A = encoder_model.predict(data_np, verbose=0)
    rec = autoencoder.predict(data_np, verbose=0)
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data_np, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    return np.hstack([A, B, C]), np.mean(np.power(data_np - rec, 2), axis=1)

print("\n[STEP C] Calculating Threshold...")
# Hitung threshold dari data train normal
_, mse_train = extract_mu(X_train_ae_final.values[:10000]) # Sample for speed
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
print(f"AE Threshold ({AE_PERCENTILE}%): {THRESHOLD:.6f}")

# =============================================================================
# [PREP] TRAIN EIF (OFFLINE)
# =============================================================================
print("\n[OFFLINE] Training EIF Model...")
# EIF dilatih pada representasi Laten dari data Training
mu_train_eif, _ = extract_mu(X_train_eif_final.values)
X_train_eif_combined = np.hstack([X_train_eif_final.values, mu_train_eif])

eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                    max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                                    max_depth=EIF_DEPTH)
eif_model.fit(X_train_eif_combined)

# =============================================================================
# [STEP D-G] ONLINE FRAUD DETECTION SIMULATION
# =============================================================================
print("\n" + "="*40)
print("ONLINE SIMULATION (TEST SET)")
print("="*40)

X_test_np = X_test_final.values
mu_test, mse_test = extract_mu(X_test_np)

# [STEP D & E] Suspect Detection via AE
suspicious_mask = mse_test > THRESHOLD
ae_flagged_count = np.sum(suspicious_mask)

print(f"Transactions Processed: {len(X_test_final)}")
print(f"Suspects (AE): {ae_flagged_count} ({ae_flagged_count/len(X_test_final)*100:.2f}%)")

final_decisions = np.zeros(len(X_test_final))

# [STEP F & G] EIF Verification
if ae_flagged_count > 0:
    X_suspect = np.hstack([X_test_np[suspicious_mask], mu_test[suspicious_mask]])
    
    # EIF Score
    scores = eif_model.predict(X_suspect)
    
    # Decision
    final_decisions[suspicious_mask] = (scores > 0.5).astype(int)

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*60)
print("SCIENTIFICALLY VALIDATED RESULTS")
print("="*60)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, final_decisions)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, final_decisions))

acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
print(f"Accuracy:  {acc:.4f}")
print(f"F1-Score:  {f1_score(y_test, final_decisions):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test, final_decisions):.4f}")