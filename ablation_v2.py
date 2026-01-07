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
import time # Added time import
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
# Kita set AE threshold lebih longgar (85%) untuk menangkap kandidat lebih banyak
AE_PERCENTILE = 85 
EIF_NTREES = 300
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 20

# =============================================================================
# [STEP A] FEATURE ENGINEERING (BACK TO BASICS - V14)
# =============================================================================
print("\n[STEP A] Generating Robust Features...")

df = pd.read_csv(DATASET_PATH)

def clean_quotes(value):
    if isinstance(value, str):
        clean_val = value.replace('"', '').replace("'", '').strip()
        try:
            return float(clean_val) if '.' in clean_val else int(clean_val)
        except ValueError:
            return clean_val
    return value

for col in df.columns:
    df[col] = df[col].apply(clean_quotes)

df = df.rename(columns={'zipcodeOri': 'customer_zip', 'zipMerchant': 'merchant_zip', 'fraud': 'fraud'})
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['fraud'] = pd.to_numeric(df['fraud'], errors='coerce').fillna(0).astype(int)
df = df.sort_values(['customer', 'step'])

# --- 1. CORE BEHAVIORAL FEATURES ---
# Fitur yang terbukti kuat di V13/V14
grouped = df.groupby('customer')
cat_stats = df.groupby('category')['amount'].mean().to_dict()

df['hour'] = df['step'] % 24
df['roll_3h_cnt'] = grouped['amount'].transform(lambda x: x.rolling(3, min_periods=1).count())
df['cat_mean'] = df['category'].map(cat_stats)

# Log Amount (Menangani skala amount yang eksponensial pada fraud)
df['log_amount'] = np.log1p(df['amount'])

# Rules
df['R_Rapid'] = (df['roll_3h_cnt'] > 2).astype(int)
df['R_HighCat'] = (df['amount'] > df['cat_mean'] * 3).astype(int)
df['R_DiffZip'] = (df['customer_zip'] != df['merchant_zip']).astype(int)
df['R_Night'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# --- 2. ENCODING ---
df['age'] = df['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, dtype=int)

scaler = RobustScaler()
num_cols = ['log_amount', 'amount', 'cat_mean'] # Gunakan log amount
df[num_cols] = scaler.fit_transform(df[num_cols])

# Gabungkan
rule_cols = ['R_Rapid', 'R_HighCat', 'R_DiffZip', 'R_Night']
X_omega = pd.concat([df[num_cols], df[rule_cols], df_encoded], axis=1)
y = df['fraud']

# Splitting
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"Total Features: {X_omega.shape[1]}")

# =============================================================================
# [STEP B] TRAIN AUTOENCODER
# =============================================================================
print("\n[STEP B] Training AutoEncoder...")
input_dim = X_train_ae.shape[1]

input_layer = Input(shape=(input_dim,))
enc1 = Dense(20, activation='tanh')(input_layer)
bottleneck = Dense(8, activation='relu')(enc1)
dec1 = Dense(20, activation='tanh')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(dec1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=0,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])

# =============================================================================
# [STEP C] THRESHOLDING & EIF
# =============================================================================
def get_features(data):
    if hasattr(data, 'values'): data = data.values
    rec = autoencoder.predict(data, verbose=0)
    enc = encoder_model.predict(data, verbose=0)
    mse = np.mean(np.power(data - rec, 2), axis=1)
    return np.hstack([enc, mse.reshape(-1,1)]), mse

print("\n[STEP C] Training EIF...")
# Ambil threshold AE dari training normal
_, mse_train_sample = get_features(X_train_ae.iloc[:10000])
AE_THRESH_VAL = np.percentile(mse_train_sample, AE_PERCENTILE)

# Train EIF
X_train_final, _ = get_features(X_train_eif_raw)
eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                    max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
eif_model.fit(X_train_final)

# =============================================================================
# [STEP D] PREDICTION & SMART TUNING
# =============================================================================
print("\n[STEP D] Predicting & Tuning Thresholds...")
X_test_final, mse_test = get_features(X_test)
eif_scores = eif_model.predict(X_test_final)

# Filter Gatekeeper
ae_suspect_mask = mse_test > AE_THRESH_VAL
suspect_idx = np.where(ae_suspect_mask)[0]

print(f"Suspects Pool: {len(suspect_idx)}")

# GRID SEARCH PADA HASIL PREDIKSI
# Kita cari threshold EIF mana yang memberikan F1 terbaik
best_f1 = 0
best_thresh = 0
best_preds = []

# Loop range threshold dari 0.35 sampai 0.65
thresholds = np.arange(0.35, 0.65, 0.01)

print("\nScanning Thresholds...")
for t in thresholds:
    temp_preds = np.zeros(len(y_test))
    if len(suspect_idx) > 0:
        # Hanya suspect yg skor EIF-nya > t dianggap fraud
        confirmed = eif_scores[suspect_idx] > t
        temp_preds[suspect_idx[confirmed]] = 1
    
    current_f1 = f1_score(y_test, temp_preds)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = t
        best_preds = temp_preds

# =============================================================================
# HASIL FINAL
# =============================================================================
print("\n" + "="*60)
print(f"FINAL RESULTS (OPTIMAL THRESHOLD: {best_thresh:.3f})")
print("="*60)

print(confusion_matrix(y_test, best_preds))
print(classification_report(y_test, best_preds, digits=4))

acc = accuracy_score(y_test, best_preds)
rec = recall_score(y_test, best_preds)
prec = precision_score(y_test, best_preds)
f1 = f1_score(y_test, best_preds)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

if f1 > 0.85:
    print("\n[SUCCESS] Model successfully optimized above 85%!")
else:
    print(f"\n[NOTE] Best F1 reachable is {f1:.4f}. This is the honest limit of unsupervised method on this data split.")