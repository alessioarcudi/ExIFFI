print("="*60)
print("IMPLEMENTASI AE-EIF+ (STRICT PAPER ARCHITECTURE & FEATURES)")
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
from sklearn.decomposition import PCA
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
# KONFIGURASI SESUAI JURNAL
# =============================================================================
# Paper: "We trained the model for 50 epochs" [cite: 855]
# Namun kita set 100 agar aman, dengan EarlyStopping
EPOCHS = 110 
BATCH_SIZE = 32
EIF_NTREES = 400 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 20 

# Threshold adjustment (Statistical/Tukey method di paper, kita pakai percentile)
AE_PERCENTILE = 93

dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# [STEP A] FEATURE SELECTION & ENGINEERING (PAPER STRICT)
# =============================================================================
print("\n[STEP A] Executing Paper-Based Feature Engineering...")
print("Ref: 'Having AMOUNT attribute... will potentially create monetary-based outliers' ")

df_banksim = pd.read_csv(dataset_path)

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
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

# --- 1. MEMBUANG AMOUNT (MONETARY) ---
# Paper Section 4.3.2: "Monetary amount can be useful... while for the data anomaly scope... this feature has no relevance." [cite: 281]
# Kita simpan Y (label), tapi X tidak boleh ada Amount.

# --- 2. CATEGORICAL FEATURES ---
# Paper menggunakan One-Hot Encoding pada fitur kategorikal [cite: 353-354]
# Di BankSim, fitur kategorikal utama adalah: Category, Age, Gender, Merchant.
# Kita gabungkan Category+Merchant sebagai proxy 'ACCOUNT_DC' di paper.

df_feat = df_banksim.copy()
df_feat['age'] = df_feat['age'].astype(str)
df_feat['merchant'] = df_feat['merchant'].astype(str)
df_feat['category'] = df_feat['category'].astype(str)
df_feat['gender'] = df_feat['gender'].astype(str)

# One-Hot Encoding
print("   -> Performing One-Hot Encoding...")
categorical_cols = ['category', 'age', 'gender', 'merchant']
X_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

print(f"   -> Dimensions after One-Hot: {X_encoded.shape[1]}")

# --- 3. DIMENSIONALITY REDUCTION (PCA) ---
# Paper Section 4.3.6: "Using PCA, we were able to significantly decrease data dimensionality" [cite: 411]
# Paper menyarankan variance explained 99% 

print("   -> Applying PCA (99% Variance Explained)...")
# PCA butuh data normalisasi/standarisasi dulu, tapi karena ini binary (0/1), paper bilang centering tidak wajib, 
# tapi Scikit-learn PCA menyarankan centered data. Kita biarkan raw one-hot (0/1).
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_encoded)

print(f"   -> Dimensions after PCA: {X_pca.shape[1]} components")

# Normalisasi MinMax (0-1) agar cocok dengan Sigmoid Output AE
scaler = MinMaxScaler()
X_final = scaler.fit_transform(X_pca)

y = df_feat['fraud']

# Splitting
X_normal = X_final[y == 0]
X_fraud = X_final[y == 1]

# 70% Train, 30% Test [cite: 438-440]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = np.vstack([X_train_eif_base, X_fraud]) # Simplifikasi test set campuran
y_test = np.concatenate([np.zeros(len(X_train_eif_base)), np.ones(len(X_fraud))])

# Sisa normal untuk EIF training
X_train_eif = X_train_eif_base

print(f"Data Splits -> AE Train: {len(X_train_ae)} | Test: {len(X_test)}")

# =============================================================================
# [STEP B] TRAINING AUTOENCODER (DEEP ARCHITECTURE)
# Ref: Figure 11 & Section 4.4.5 
# Encoder: Input -> 128 -> 64 -> 16 -> 8
# Decoder: 8 -> 16 -> 64 -> 128 -> Output
# =============================================================================
print("\n[STEP B] Training Deep Autoencoder (Paper Architecture)...")

input_dim = X_train_ae.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder 
enc1 = Dense(128, activation='relu')(input_layer)
enc2 = Dense(64, activation='relu')(enc1)
enc3 = Dense(16, activation='relu')(enc2)
bottleneck = Dense(8, activation='relu', name='bottleneck')(enc3) # Latent Space = 8

# Decoder 
dec1 = Dense(16, activation='relu')(bottleneck)
dec2 = Dense(64, activation='relu')(dec1)
dec3 = Dense(128, activation='relu')(dec2)
output_layer = Dense(input_dim, activation='sigmoid')(dec3) # Sigmoid output

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

# Paper: "Binary_crossentropy loss and adam optimizer" [cite: 679] (Mentioned in Supervised, implied for AE reconstruction of binary/scaled data)
autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse') 

# Paper: 50 epochs[cite: 855], we use 100 with EarlyStopping
autoencoder.fit(X_train_ae, X_train_ae, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE,
                shuffle=True, 
                validation_split=0.1, 
                verbose=1,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

# =============================================================================
# [STEP C] THRESHOLDING
# =============================================================================
def extract_features(data_np):
    # Latent (A), Reconstruction Error (MSE), Cosine (C)
    A = encoder_model.predict(data_np, verbose=0)
    rec = autoencoder.predict(data_np, verbose=0)
    
    mse_loss = np.mean(np.power(data_np - rec, 2), axis=1)
    
    # Distance & Cosine for EIF
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data_np, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    
    return np.hstack([A, B, C]), mse_loss

print("\n[STEP C] Establishing Threshold...")
_, mse_train = extract_features(X_train_ae[:10000])
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
print(f"AE Threshold ({AE_PERCENTILE}%): {THRESHOLD:.6f}")

# =============================================================================
# [OFFLINE] TRAINING EIF (HYBRID)
# =============================================================================
print("\n[OFFLINE] Training EIF on Latent Features...")
mu_train_eif, _ = extract_features(X_train_eif)
# Paper doesn't explicitly mention concatenating Original+Latent for EIF, usually Latent is enough.
# But for EIF+, rich features help. We stick to Latent+Error features (Mu).
X_train_eif_final = mu_train_eif 

eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                    max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_final.shape[0]), 
                                    max_depth=EIF_DEPTH)
eif_model.fit(X_train_eif_final)

# =============================================================================
# [ONLINE] DETECTION
# =============================================================================
print("\n[ONLINE] Simulating Detection...")
mu_test, mse_test = extract_features(X_test)

# AE Filter
suspicious_mask = mse_test > THRESHOLD
ae_flagged_count = np.sum(suspicious_mask)
print(f"Suspects Found (AE): {ae_flagged_count} ({ae_flagged_count/len(X_test)*100:.2f}%)")

final_decisions = np.zeros(len(X_test))

if ae_flagged_count > 0:
    X_suspect = mu_test[suspicious_mask] # Only Latent features for EIF
    scores = eif_model.predict(X_suspect)
    final_decisions[suspicious_mask] = (scores > 0.5).astype(int)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS (PAPER STRICT)")
print("="*60)
print(confusion_matrix(y_test, final_decisions))
print(classification_report(y_test, final_decisions))

acc = accuracy_score(y_test, final_decisions)
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1_score(y_test, final_decisions):.4f}")
print(f"MCC:      {matthews_corrcoef(y_test, final_decisions):.4f}")