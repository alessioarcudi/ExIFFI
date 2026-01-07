print("="*60)
print("IMPLEMENTASI AE-EIF+ (PURE BEHAVIORAL RULES - JOURNAL STRICT)")
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
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI
# =============================================================================
EPOCHS = 110 
BATCH_SIZE = 512
AE_PERCENTILE = 90 # Sedikit dilonggarkan agar Recall naik (menangkap lebih banyak fraud)
EIF_NTREES = 512 
EIF_SAMPLE_SIZE = 512
EIF_DEPTH = 32

dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# [STEP A] RULE-BASED FEATURE ENGINEERING (PURE BEHAVIOR)
# Fokus: Membuat aturan "Apakah transaksi ini aneh buat customer INI?"
# =============================================================================
print("\n[STEP A] Executing Rule-Based Feature Engineering (Pure Behavior)...")

df_banksim = pd.read_csv(dataset_path)

# Cleaning Standard
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

# --- MEMBUAT FITUR ATURAN (RULES) ---
df_feat = df_banksim.copy()

# 1. TIME RULES
# Asumsi: Step = Jam. Fraud sering terjadi di jam tidak wajar (misal dini hari)
df_feat['hour_of_day'] = df_feat['step'] % 24
# Rule: Transaksi di jam "rawan" (misal jam 0 - 6 pagi)
df_feat['R_NightTime'] = df_feat['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# 2. LOCATION RULES (Explicit di Jurnal)
df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

# 3. AMOUNT RULES (Explicit di Jurnal)
# Bandingkan amount transaksi ini dengan rata-rata customer tersebut
cust_avg_amt = df_feat.groupby('customer')['amount'].transform('mean')
# Rule: Apakah transaksi ini > 3x lipat kebiasaan dia?
df_feat['R_Spike_Spend'] = (df_feat['amount'] > (cust_avg_amt * 3)).astype(int)
# Rule: Apakah amount > 95 percentile populasi?
threshold_global = df_feat['amount'].quantile(0.95)
df_feat['R_HighAmt'] = (df_feat['amount'] > threshold_global).astype(int)

# 4. FREQUENCY RULES (Explicit di Jurnal)
# Hitung berapa kali transaksi dalam step (jam) yang sama
step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='tx_per_step')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
# Rule: Lebih dari 1 transaksi dalam 1 jam?
df_feat['R_RapidFire'] = (df_feat['tx_per_step'] > 1).astype(int)

# 5. CATEGORY & PROFILE (One Hot)
# Ini agar input layer mendekati 38 nodes seperti di jurnal
df_feat['age'] = df_feat['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

# --- FINAL FEATURES ---
# Kita gabungkan Rules (Binary) + Raw Scaled Data + One Hot
scaler = MinMaxScaler()
df_feat['Amount_Scaled'] = scaler.fit_transform(df_feat[['amount']])

# Fitur yang masuk ke AE
rule_cols = ['R_NightTime', 'R_DiffZip', 'R_Spike_Spend', 'R_HighAmt', 'R_RapidFire', 'Amount_Scaled']
X_omega = pd.concat([df_feat[rule_cols], df_encoded], axis=1)
y = df_feat['fraud']

print(f"Features Generated: {X_omega.shape[1]} (Target: ~38 nodes)")

# Splitting Strict
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# Train (Normal Only) vs Test (Mixed)
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"AE Train Size: {len(X_train_ae)} | Test Size: {len(X_test)}")

# =============================================================================
# [STEP B] TRAINING AUTOENCODER (38 Nodes Architecture)
# =============================================================================
print("\n[STEP B] Training AutoEncoder...")

input_dim = X_train_ae.shape[1]
# Paper: 38 -> 14 -> 7 -> 14 -> 38
input_layer = Input(shape=(input_dim,))
encoder1 = Dense(14, activation='relu')(input_layer)
bottleneck = Dense(7, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(14, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoder1) # Sigmoid karena input 0-1 (Rules/OneHot)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[EarlyStopping(patience=15)])

# =============================================================================
# [STEP C] THRESHOLDING
# =============================================================================
def extract_mu(data_np):
    if hasattr(data_np, 'values'): data_np = data_np.values.astype(float)
    A = encoder_model.predict(data_np, verbose=0)
    rec = autoencoder.predict(data_np, verbose=0)
    # Euclidean Distance
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    # Cosine
    norm_d = np.linalg.norm(data_np, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    return np.hstack([A, B, C]), np.mean(np.power(data_np - rec, 2), axis=1)

print("\n[STEP C] Establishing Threshold...")
_, mse_train = extract_mu(X_train_ae.iloc[:10000])
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
print(f"AE Threshold ({AE_PERCENTILE}%): {THRESHOLD:.6f}")

# =============================================================================
# [OFFLINE] TRAINING EIF
# =============================================================================
print("\n[OFFLINE] Training EIF...")
mu_train_eif, _ = extract_mu(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                    max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                                    max_depth=EIF_DEPTH)
eif_model.fit(X_train_eif_combined)

# =============================================================================
# [ONLINE] DETECTION
# =============================================================================
print("\n[ONLINE] Simulating Detection...")
X_test_np = X_test.values
mu_test, mse_test = extract_mu(X_test_np)

# AE Filter
suspicious_mask = mse_test > THRESHOLD
ae_flagged_count = np.sum(suspicious_mask)

final_decisions = np.zeros(len(X_test))

if ae_flagged_count > 0:
    X_suspect = np.hstack([X_test_np[suspicious_mask], mu_test[suspicious_mask]])
    scores = eif_model.predict(X_suspect)
    # Decision
    final_decisions[suspicious_mask] = (scores > 0.5).astype(int)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*60)
print("FINAL PURE BEHAVIORAL RESULTS")
print("="*60)
print(confusion_matrix(y_test, final_decisions))
print(classification_report(y_test, final_decisions))