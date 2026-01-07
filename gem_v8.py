print("="*60)
print("KOMPARASI ILMIAH: AE vs AE-EIF vs AE-EIF+ (JALUR A)")
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
from sklearn.preprocessing import MinMaxScaler
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
# KONFIGURASI (Sesuai Paper Dataset 2)
# =============================================================================
EPOCHS = 110 
BATCH_SIZE = 64
EIF_NTREES = 200 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 18 

# Threshold AE (Kita coba 95% sesuai 'Winner' sebelumnya untuk Presisi)
AE_PERCENTILE = 95 

# Paths
dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
ae_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/ae_38nodes.keras'
enc_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/enc_38nodes.keras'

# =============================================================================
# 1. LOAD DATA & CLEANING
# =============================================================================
print(f"Loading Dataset...")
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
df_banksim['age'] = df_banksim['age'].astype(str) # Keep as category for One-Hot
df_banksim['amount'] = pd.to_numeric(df_banksim['amount'], errors='coerce')
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

# =============================================================================
# 2. FEATURE ENGINEERING (ONE-HOT STRATEGY -> TARGET 38 NODES)
# =============================================================================
print("Executing Feature Engineering (One-Hot Strategy)...")
df_feat = df_banksim.copy()

# A. Numerical Features (Scaled)
scaler = MinMaxScaler()
df_feat['Amount_Scaled'] = scaler.fit_transform(df_feat[['amount']])

# B. Rule-Based Binary Features (Sesuai Paper)
# 1. High Amount (Binary)
threshold_amt = df_feat['amount'].quantile(0.95)
df_feat['R_HighAmount'] = (df_feat['amount'] > threshold_amt).astype(int)

# 2. Different Zip (Binary)
df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

# 3. Transaction Frequency (Simulated by Step count per user)
step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='step_tx_count')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
df_feat['R_HighFreq'] = (df_feat['step_tx_count'] > 1).astype(int)

# C. Categorical Features -> One-Hot Encoding
# Ini kunci untuk mendapatkan ~38 features
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

# Gabungkan Fitur
# Fitur Numerik + Rules + One-Hot
feature_cols = ['Amount_Scaled', 'step', 'R_HighAmount', 'R_DiffZip', 'R_HighFreq']
X_omega = pd.concat([df_feat[feature_cols], df_encoded], axis=1)
y = df_feat['fraud']

print(f"Feature Dimension (Nodes): {X_omega.shape[1]}") 
# Harapannya mendekati 38 (misal: 5 numeric + 15 category + 8 age + 4 gender = ~32-35)

# Splitting
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"Data Splits -> AE Train: {len(X_train_ae)}, Test: {len(X_test)}")

# =============================================================================
# 3. TRAIN AUTOENCODER (ARSITEKTUR 38-14-7-14-38)
# =============================================================================
input_dim = X_train_ae.shape[1] 
print(f"AE Input Layer: {input_dim} nodes (Paper says 38)")

input_layer = Input(shape=(input_dim,))
# Paper: "trio of hidden layers of 14, 7, and 14 nodes"
encoder1 = Dense(14, activation='relu')(input_layer) 
bottleneck = Dense(7, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(14, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoder1) # Sigmoid karena One-Hot/MinMax (0-1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck) 
autoencoder.compile(optimizer='adam', loss='mse')

if os.path.exists(ae_path):
    # Cek dimensi model lama, jika beda harus retrain
    loaded_model = load_model(ae_path)
    if loaded_model.input_shape[1] != input_dim:
        print("Dimension mismatch with saved model. Retraining...")
        autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        shuffle=True, validation_split=0.1, verbose=1,
                        callbacks=[EarlyStopping(patience=15)])
        autoencoder.save(ae_path)
        encoder_model.save(enc_path)
    else:
        print("Loading compatible pre-trained AutoEncoder...")
        autoencoder = loaded_model
        encoder_model = load_model(enc_path)
else:
    print(f"\nTraining AutoEncoder ({EPOCHS} Epochs)...")
    autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    shuffle=True, validation_split=0.1, verbose=1,
                    callbacks=[EarlyStopping(patience=15)])
    autoencoder.save(ae_path)
    encoder_model.save(enc_path)

# =============================================================================
# 4. PRE-CALCULATION
# =============================================================================
def extract_mu(data):
    if hasattr(data, 'values'): data = data.values.astype(float)
    A = encoder_model.predict(data, verbose=0)
    rec = autoencoder.predict(data, verbose=0)
    B = np.linalg.norm(data - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    # Epsilon kecil untuk hindari div by zero
    C = (np.sum(data * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    return np.hstack([A, B, C]), np.mean(np.power(data - rec, 2), axis=1)

print("\nExtracting Features (Normal Only for Threshold)...")
_, mse_train_ae = extract_mu(X_train_ae.iloc[:10000]) 
THRESHOLD = np.percentile(mse_train_ae, AE_PERCENTILE)
print(f"AE Threshold ({AE_PERCENTILE}%): {THRESHOLD:.6f}")

print("Extracting Features for EIF Training...")
mu_train_eif, _ = extract_mu(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

print("Extracting Features for Test Set...")
mu_test, mse_test = extract_mu(X_test)
X_test_np = X_test.values

# =============================================================================
# 5. EVALUASI KOMPARASI
# =============================================================================
results = {}

# --- A. AE ONLY ---
ae_preds = (mse_test > THRESHOLD).astype(int)
results['AE Only'] = {
    'acc': accuracy_score(y_test, ae_preds),
    'f1': f1_score(y_test, ae_preds),
    'prec': precision_score(y_test, ae_preds),
    'rec': recall_score(y_test, ae_preds)
}

# --- B. AE-EIF+ (Hybrid) ---
print("\nTraining AE-EIF+...")
eif = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                              max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                              max_depth=EIF_DEPTH)
eif.fit(X_train_eif_combined)

# Predict
suspicious_mask = mse_test > THRESHOLD
final_preds = np.zeros(len(X_test))

if np.sum(suspicious_mask) > 0:
    X_suspect = np.hstack([X_test_np[suspicious_mask], mu_test[suspicious_mask]])
    scores = eif.predict(X_suspect)
    # Default Cutoff 0.5
    final_preds[suspicious_mask] = (scores > 0.5).astype(int)

results['AE-EIF+'] = {
    'acc': accuracy_score(y_test, final_preds),
    'f1': f1_score(y_test, final_preds),
    'prec': precision_score(y_test, final_preds),
    'rec': recall_score(y_test, final_preds)
}

# --- REPORT ---
print("\n" + "="*80)
print(f"{'MODEL':<15} | {'ACCURACY':<10} | {'F1-SCORE':<10} | {'PRECISION':<10} | {'RECALL':<10}")
print("="*80)
for name, m in results.items():
    print(f"{name:<15} | {m['acc']:.4f}     | {m['f1']:.4f}     | {m['prec']:.4f}     | {m['rec']:.4f}")
print("="*80)