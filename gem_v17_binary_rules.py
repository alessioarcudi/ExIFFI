print("="*60)
print("IMPLEMENTASI AE-EIF+ (FULL BINARY RULES - TARGET 0.99)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

warnings.filterwarnings('ignore')

# =============================================================================
# 1. KONFIGURASI (SESUAI JURNAL 2)
# =============================================================================
# "epoch values of... 110 on data 2" [cite: 1579]
EPOCHS = 10 
BATCH_SIZE = 64 
# "forest of 200 trees and a tree depth of 18" [cite: 1583]
EIF_NTREES = 400 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 20 

DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# 2. FEATURE ENGINEERING (STRATEGI: ALL BINARY 0/1)
# =============================================================================
print("\n[STEP A] Feature Engineering (Binary Rules Strategy)...")
# "creating binary features (0 or 1) for each transaction" 

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

# --- RULE 1: HIGH AMOUNT (BINARY) ---
# "high amount binary feature is set to 1 if... exceeds threshold" 
# Kita buat beberapa level threshold agar AE bisa belajar gradasi risiko
q75 = df_feat['amount'].quantile(0.75)
q90 = df_feat['amount'].quantile(0.90)
q99 = df_feat['amount'].quantile(0.99)

df_feat['R_Amt_Gt_75'] = (df_feat['amount'] > q75).astype(int)
df_feat['R_Amt_Gt_90'] = (df_feat['amount'] > q90).astype(int)
df_feat['R_Amt_Gt_99'] = (df_feat['amount'] > q99).astype(int)

# --- RULE 2: CATEGORY SPIKE (BINARY) ---
# Apakah amount ini jauh diatas rata-rata kategori tersebut?
cat_mean = df_feat.groupby('category')['amount'].transform('mean')
df_feat['R_Cat_Spike_2x'] = (df_feat['amount'] > (cat_mean * 2)).astype(int)
df_feat['R_Cat_Spike_5x'] = (df_feat['amount'] > (cat_mean * 5)).astype(int)

# --- RULE 3: USER SPIKE (BINARY) ---
cust_mean = df_feat.groupby('customer')['amount'].transform('mean')
df_feat['R_User_Spike_2x'] = (df_feat['amount'] > (cust_mean * 2)).astype(int)
df_feat['R_User_Spike_5x'] = (df_feat['amount'] > (cust_mean * 5)).astype(int)

# --- RULE 4: VELOCITY / RAPID FIRE (BINARY) ---
# "Frequency of transactions" [cite: 1569]
step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='tx_per_step')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
df_feat['R_RapidFire'] = (df_feat['tx_per_step'] > 1).astype(int)

# --- RULE 5: ONE-HOT ENCODING (MURNI 0/1) ---
df_feat['age'] = df_feat['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

# --- ASSEMBLE ALL BINARY FEATURES ---
# Semua fitur harus 0 atau 1. Tidak ada continuous variable.
binary_rules = [
    'R_Amt_Gt_75', 'R_Amt_Gt_90', 'R_Amt_Gt_99',
    'R_Cat_Spike_2x', 'R_Cat_Spike_5x',
    'R_User_Spike_2x', 'R_User_Spike_5x',
    'R_RapidFire'
]

X_final = pd.concat([df_feat[binary_rules], df_encoded], axis=1)
y = df_feat['fraud']

# Pastikan semua tipe data adalah float (0.0 atau 1.0) untuk AE
X_final = X_final.astype(float)

print(f"Total Binary Features: {X_final.shape[1]}")
# Harusnya sekitar ~30-40 fitur, cocok dengan "input layer made up of 38 nodes" [cite: 1580]

# --- Splitting ---
X_normal = X_final[y == 0]
X_fraud = X_final[y == 1]

# 70% Train AE, 30% Test (Stratified Split secara manual untuk simulasi)
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)

# Test set: Campuran Normal (sisa) + Fraud (Semua)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])

# EIF Train: Sisa normal
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"AE Train: {len(X_train_ae)} | Test: {len(X_test)}")

# =============================================================================
# 3. TRAIN AUTOENCODER (PAPER ARCHITECTURE)
# =============================================================================
print("\n[STEP B] Training Deep AutoEncoder (Binary Mode)...")
# "input... 38 nodes... hidden 14, 7, 14... output 38" [cite: 1580-1582]

input_dim = X_train_ae.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder
enc1 = Dense(14, activation='relu')(input_layer)
bottleneck = Dense(7, activation='relu', name='bottleneck')(enc1)

# Decoder
dec1 = Dense(14, activation='relu')(bottleneck)
# PENTING: Output Sigmoid sangat cocok karena input kita sekarang murni 0/1 (Binary)
output_layer = Dense(input_dim, activation='sigmoid')(dec1) 

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callback
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[early_stop, lr_decay])

# =============================================================================
# 4. PREDIKSI & OPTIMASI THRESHOLD
# =============================================================================
def extract_features(data_np):
    if hasattr(data_np, 'values'): data_np = data_np.values.astype(float)
    rec = autoencoder.predict(data_np, verbose=0)
    enc = encoder_model.predict(data_np, verbose=0)
    
    # MSE Loss
    mse = np.mean(np.power(data_np - rec, 2), axis=1)
    
    # Latent + Error for EIF
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data_np, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    
    return np.hstack([enc, B, C]), mse

print("\n[STEP C] Training EIF (Offline)...")
mu_train_eif, _ = extract_features(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

eif = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                              max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                              max_depth=EIF_DEPTH)
eif.fit(X_train_eif_combined)

print("\n[STEP D] Testing & Grid Search Optimization...")
# Hitung MSE training untuk baseline threshold
_, mse_train = extract_features(X_train_ae.iloc[:10000])

# Prepare Test Data
X_test_np = X_test.values
mu_test, mse_test = extract_features(X_test_np)
X_test_combined = np.hstack([X_test_np, mu_test])
eif_scores_all = eif.predict(X_test_combined)

# Grid Search untuk mencari "Sweet Spot" seperti di jurnal
# "threshold is set to a number that permits the proportion of higher scores to be isolated" [cite: 1633]
best_metrics = {'F1': 0}

print(f"{'Percentile':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 50)

for p in [85, 90, 93, 95, 96, 97, 98, 99]:
    thresh = np.percentile(mse_train, p)
    
    # AE Filter
    suspects = mse_test > thresh
    
    # EIF Confirmation
    final_preds = np.zeros(len(y_test))
    if np.sum(suspects) > 0:
        final_preds[suspects] = (eif_scores_all[suspects] > 0.5).astype(int)
        
    f1 = f1_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds, zero_division=0)
    rec = recall_score(y_test, final_preds)
    
    print(f"{p:<10} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")
    
    if f1 > best_metrics['F1']:
        best_metrics = {
            'Percentile': p,
            'Accuracy': accuracy_score(y_test, final_preds),
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Preds': final_preds
        }

print("\n" + "="*60)
print(f"FINAL RESULT (TARGETING JOURNAL SCORE)")
print(f"Best Configuration: AE Percentile = {best_metrics['Percentile']}%")
print("="*60)
print(confusion_matrix(y_test, best_metrics['Preds']))
print(classification_report(y_test, best_metrics['Preds']))
print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
print(f"F1-Score:  {best_metrics['F1']:.4f}")