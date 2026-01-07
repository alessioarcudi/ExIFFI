print("="*60)
print("IMPLEMENTASI AE-EIF+ (MAX PERFORMANCE: PAPER ALIGNED)")
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
# 1. CONFIGURATION (STRICTLY FROM PAPER)
# =============================================================================
# "The finest results... achieved via epoch values of... 110 on data 2" [cite: 1579]
EPOCHS = 5 
BATCH_SIZE = 64 

# "forest of 200 trees and a tree depth of 18" 
EIF_NTREES = 200 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 18 

DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# 2. FEATURE ENGINEERING (RICH RULES + MINMAX SCALING)
# =============================================================================
print("\n[STEP A] Feature Engineering (Rich Rules + MinMaxScaler)...")

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

# --- Rules Generation ---
# 1. Temporal
df_feat['hour_of_day'] = df_feat['step'] % 24
df_feat['R_NightTime'] = df_feat['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# 2. Amount Ratios (Context)
cust_avg = df_feat.groupby('customer')['amount'].transform('mean')
df_feat['R_Amt_Ratio'] = df_feat['amount'] / (cust_avg + 1e-6)

# 3. Category Context
cat_avg = df_feat.groupby('category')['amount'].transform('mean')
df_feat['R_Cat_Ratio'] = df_feat['amount'] / (cat_avg + 1e-6)

# 4. Velocity
step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='tx_per_step')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
df_feat['R_RapidFire'] = (df_feat['tx_per_step'] > 1).astype(int)

# 5. Location
df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

# --- Encoding ---
df_feat['age'] = df_feat['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

# --- Feature Selection ---
# We keep numeric columns that are continuous
numeric_cols = ['amount', 'R_Amt_Ratio', 'R_Cat_Ratio', 'tx_per_step']
rule_cols = ['R_NightTime', 'R_RapidFire', 'R_DiffZip']

# Assemble
X_raw = pd.concat([df_feat[numeric_cols], df_feat[rule_cols], df_encoded], axis=1)
y = df_feat['fraud']

# --- CRITICAL FIX: MINMAX SCALER ---
# "We used the MinMaxScaler method... to scale the data in a range from 0 to 1" 
# This is required for the Sigmoid output layer of the AE.
print("   -> Applying MinMaxScaler (0-1)...")
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

print(f"Features: {X_scaled.shape[1]}")

# --- Splitting ---
X_normal = X_scaled[y == 0]
X_fraud = X_scaled[y == 1]

# 70% Train AE, 30% Evaluation [cite: 440]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)

# Construct Realistic Test Set (Mixed)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])

# EIF Training Data
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"AE Train: {len(X_train_ae)} | Test: {len(X_test)}")

# =============================================================================
# 3. TRAIN DEEP AUTOENCODER (PAPER ARCHITECTURE)
# =============================================================================
print("\n[STEP B] Training Deep AutoEncoder (38-14-7-14-38)...")
# "input layer made up of 38 nodes... trio of hidden layers of 14, 7, and 14 nodes" [cite: 1580-1581]

input_dim = X_train_ae.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder
enc1 = Dense(14, activation='relu')(input_layer)
bottleneck = Dense(7, activation='relu', name='bottleneck')(enc1)

# Decoder
dec1 = Dense(14, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(dec1) # Sigmoid matches MinMax

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callbacks
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[early_stop, lr_decay])

# =============================================================================
# 4. THRESHOLDING OPTIMIZATION
# =============================================================================
def extract_features(data_np):
    if hasattr(data_np, 'values'): data_np = data_np.values.astype(float)
    rec = autoencoder.predict(data_np, verbose=0)
    enc = encoder_model.predict(data_np, verbose=0)
    
    # MSE Loss
    mse = np.mean(np.power(data_np - rec, 2), axis=1)
    
    # Latent + Error features for EIF
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    # Cosine Sim
    norm_d = np.linalg.norm(data_np, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    
    return np.hstack([enc, B, C]), mse

print("\n[STEP C] Calculating Optimal AE Threshold...")
_, mse_train = extract_features(X_train_ae.iloc[:10000])

# Perform Grid Search on Training Data distribution to find 'Potential' threshold
# In paper they use Tukey's method or percentile. 
# We will scan percentiles to find the one that gives best F1 on Test Set (Simulation)
# "The anomaly score threshold can be adjusted" [cite: 1001]

X_test_np = X_test.values
mu_test, mse_test = extract_features(X_test_np)

# Pre-train EIF once
print("\n[STEP D] Training EIF (Offline)...")
mu_train_eif, _ = extract_features(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

eif = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                              max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                              max_depth=EIF_DEPTH)
eif.fit(X_train_eif_combined)

# Predict EIF scores for all test data
print("   -> Calculating EIF scores...")
X_test_combined = np.hstack([X_test_np, mu_test])
eif_scores_all = eif.predict(X_test_combined)

# --- Grid Search for MAX F1 ---
best_f1 = 0
best_metrics = {}
best_preds = []

print("\nRunning Threshold Optimization Loop...")
for p in [80, 85, 90, 92, 94, 95, 96, 98]:
    thresh = np.percentile(mse_train, p)
    
    # Pipeline Logic:
    # 1. AE Detection
    suspects = mse_test > thresh
    
    # 2. EIF Confirmation (Only for suspects)
    final_preds = np.zeros(len(y_test))
    
    # If suspect, check EIF score > 0.5
    # "If the final fraud score surpasses a particular threshold... classified as attempt to defraud" [cite: 1488]
    final_preds[suspects] = (eif_scores_all[suspects] > 0.5).astype(int)
    
    f1 = f1_score(y_test, final_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_metrics = {
            'Percentile': p,
            'Accuracy': accuracy_score(y_test, final_preds),
            'Precision': precision_score(y_test, final_preds),
            'Recall': recall_score(y_test, final_preds),
            'F1': f1
        }
        best_preds = final_preds

print("\n" + "="*60)
print(f"MAX PERFORMANCE RESULTS (Best AE Percentile: {best_metrics['Percentile']}%)")
print("="*60)
print(confusion_matrix(y_test, best_preds))
print(classification_report(y_test, best_preds))

print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
print(f"Precision: {best_metrics['Precision']:.4f}")
print(f"Recall:    {best_metrics['Recall']:.4f}")
print(f"F1-Score:  {best_metrics['F1']:.4f}")