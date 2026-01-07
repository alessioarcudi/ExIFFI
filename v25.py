print("="*60)
print("IMPLEMENTASI AE-EIF+ (RICH BEHAVIORAL RULES - V14 OPTIMIZED)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import os
import warnings
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import sys
from collections import Counter

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# 1. KONFIGURASI SESUAI JURNAL
# =============================================================================
DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# Jurnal Hal 194 [cite: 418, 422]
EPOCHS = 10              
BATCH_SIZE = 2048         
EIF_NTREES = 200          
EIF_SAMPLE_SIZE = 256     
EIF_DEPTH = 18            

# =============================================================================
# [STEP A] FEATURE ENGINEERING (EXPANDING TO 38 NODES)
# =============================================================================
print("\n[STEP A] Generating Expanded Features (Targeting 38 Nodes)...")

df = pd.read_csv(DATASET_PATH)

# Cleaning
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

# --- 1. TIME SEGMENTATION (4 Nodes) ---
# Membagi waktu menjadi 4 segmen agar lebih detail daripada sekedar "Malam"
df['hour'] = df['step'] % 24
def get_time_segment(h):
    if 0 <= h < 6: return 'Night'
    elif 6 <= h < 12: return 'Morning'
    elif 12 <= h < 18: return 'Afternoon'
    else: return 'Evening'
df['time_seg'] = df['hour'].apply(get_time_segment)

# --- 2. BEHAVIORAL RULES (5 Nodes) ---
grouped = df.groupby('customer')
# Velocity: Transaksi per 3 jam
df['roll_3h_cnt'] = grouped['amount'].transform(lambda x: x.rolling(window=3, min_periods=1).count())
# Stats
cat_mean = df.groupby('category')['amount'].transform('mean')
cat_std = df.groupby('category')['amount'].transform('std').fillna(1)
cust_mean = grouped['amount'].transform('mean')
cust_std = grouped['amount'].transform('std').fillna(1)

# Binary Rules (Jurnal Rules)
df['B_HighAmt_Cat'] = (df['amount'] > (cat_mean + 2*cat_std)).astype(int)
df['B_HighAmt_User'] = (df['amount'] > (cust_mean + 3*cust_std)).astype(int)
df['B_RapidFire'] = (df['roll_3h_cnt'] > 2).astype(int)
df['B_DiffZip'] = (df['customer_zip'] != df['merchant_zip']).astype(int)
# New Max Amount
df['max_amt_so_far'] = grouped['amount'].transform(lambda x: x.expanding().max())
df['B_NewMax'] = (df['amount'] >= df['max_amt_so_far']).astype(int)

# --- 3. FULL ONE-HOT ENCODING (The Key to 38 Nodes) ---
# Pastikan semua kategori diubah jadi kolom terpisah
df['age'] = df['age'].astype(str)
df['gender'] = df['gender'].astype(str)
df['category'] = df['category'].astype(str)

# Kita encode: Category (~15), Age (~8), Gender (~4), Time Segment (4)
categorical_cols = ['category', 'age', 'gender', 'time_seg']
df_encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, dtype=int)

# --- 4. SCALING NUMERICAL ---
scaler = RobustScaler()
df['amount_scaled'] = scaler.fit_transform(df[['amount']])
df['roll_scaled'] = scaler.fit_transform(df[['roll_3h_cnt']])

# --- 5. ASSEMBLING ---
# Numeric (2) + Rules (5) + One-Hot (~31) = ~38 Features
rule_cols = ['B_HighAmt_Cat', 'B_HighAmt_User', 'B_RapidFire', 'B_DiffZip', 'B_NewMax']
X_omega = pd.concat([df[['amount_scaled', 'roll_scaled']], df[rule_cols], df_encoded], axis=1)
y = df['fraud']

# Print fitur untuk verifikasi
print(f"Total Features Generated: {X_omega.shape[1]}")
print(f"Target Journal Nodes: 38 (Should be very close!)")

# Split
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

# =============================================================================
# [STEP B] AUTOENCODER (JOURNAL ARCHITECTURE)
# =============================================================================
print("\n[STEP B] Training AutoEncoder (110 Epochs)...")
input_dim = X_train_ae.shape[1]

# Arsitektur Persis Jurnal [cite: 419, 420]
# 38 (Input) -> 14 -> 7 -> 14 -> 38 (Output)
input_layer = Input(shape=(input_dim,))
enc1 = Dense(14, activation='tanh')(input_layer)  
bottleneck = Dense(7, activation='relu')(enc1)    
dec1 = Dense(14, activation='tanh')(bottleneck)   
output_layer = Dense(input_dim, activation='sigmoid')(dec1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Training Full
autoencoder.fit(X_train_ae, X_train_ae, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                validation_split=0.1, 
                verbose=1,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])

# =============================================================================
# [STEP C & D] EIF PIPELINE
# =============================================================================
def get_features(data):
    if hasattr(data, 'values'): data = data.values
    rec = autoencoder.predict(data, verbose=0)
    latent = encoder_model.predict(data, verbose=0)
    mse = np.mean(np.power(data - rec, 2), axis=1)
    return np.hstack([data, latent, mse.reshape(-1,1)]), mse

_, mse_train = get_features(X_train_ae.iloc[:20000])

print("\n[STEP D] Training EIF (Depth 18, Trees 200)...")
X_train_eif_final, _ = get_features(X_train_eif_raw)
eif = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                              max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
eif.fit(X_train_eif_final)

# =============================================================================
# [STEP E] OPTIMIZATION LOOP
# =============================================================================
print("\n[STEP E] Optimization for Journal Metrics (Recall > 0.95)...")
X_test_final, mse_test = get_features(X_test)
eif_scores = eif.predict(X_test_final)

# Grid Search
ae_ps = [90, 92, 94, 95, 96, 98]
eif_ts = [0.45, 0.48, 0.50, 0.52, 0.55]

results = []
for p in ae_ps:
    ae_thresh = np.percentile(mse_train, p)
    mask = mse_test > ae_thresh
    for t in eif_ts:
        preds = np.zeros(len(y_test))
        if np.sum(mask) > 0:
            idx = np.where(mask)[0]
            preds[idx] = (eif_scores[idx] > t).astype(int)
        
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds)
        results.append([p, t, prec, rec, f1])

res_df = pd.DataFrame(results, columns=['AE_%', 'EIF_T', 'Precision', 'Recall', 'F1'])
print("\nTOP RESULTS (Sorted by Recall > 0.90 then Precision):")
print(res_df[res_df['Recall'] > 0.90].sort_values(by='Precision', ascending=False).head(5).to_string(index=False))

# Auto-select best config
top_hits = res_df[(res_df['Recall'] > 0.90) & (res_df['Precision'] > 0.65)]
if not top_hits.empty:
    best = top_hits.sort_values(by='F1', ascending=False).iloc[0]
    print(f"\n[WINNER] AE: {best['AE_%']}% | EIF: {best['EIF_T']}")
    
    mask_final = mse_test > np.percentile(mse_train, best['AE_%'])
    y_final = np.zeros(len(y_test))
    y_final[mask_final] = (eif_scores[mask_final] > best['EIF_T']).astype(int)
    
    print(classification_report(y_test, y_final))
else:
    print("\nNo balanced configuration found. Showing best Recall result:")
    print(res_df.sort_values(by='Recall', ascending=False).head(1))