print("="*60)
print("IMPLEMENTASI AE-EIF+ (Paper Compliance Version - Final)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
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

# CONFIGURATION
# Optimized according to Journal for Dataset 2
EPOCHS = 110  #  "epoch values of... 110 on data 2"
BATCH_SIZE = 64
EIF_NTREES = 200 # [cite: 422] "forest of 200 trees"
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 18 # [cite: 422] "tree depth of 18"

# PARAMETER FOR OPTIMIZATION
# Lowering this lets more "potential" frauds pass to EIF, improving Recall
# Paper implies a threshold that isolates "higher scores" 
AE_PERCENTILE_THRESHOLD = 85

# PATHS
dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
model_save_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/ae_model_optimized.keras'
encoder_save_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/encoder_model_optimized.keras'

# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================
print(f"Loading Dataset...")
df_banksim = pd.read_csv(dataset_path)

def clean_quotes(value):
    if isinstance(value, str):
        value = value.replace('"', '').replace("'", '').strip()
        try:
            return float(value) if '.' in value else int(value)
        except:
            return value
    return value

df_clean = df_banksim.copy()
for col in df_clean.columns:
    df_clean[col] = df_clean[col].apply(clean_quotes)

df_clean = df_clean.rename(columns={
    'zipcodeOri': 'customer_zip',
    'zipMerchant': 'merchant_zip',
    'fraud': 'fraud'
})

df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce').fillna(0).astype(int)
df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
df_clean['fraud'] = pd.to_numeric(df_clean['fraud'], errors='coerce').fillna(0).astype(int)

# =============================================================================
# 2. RULE-BASED FEATURE ENGINEERING (PAPER SECTION 6.3.2)
# =============================================================================
print("\nExecuting Rule-Based Feature Engineering...")

def create_paper_features(df):
    df_feat = df.copy()
    
    # 1. Binary Rule: High Amount [cite: 339]
    # "high amount... if transaction amount exceeds a specific threshold"
    threshold_amt = df_feat[df_feat['fraud']==0]['amount'].quantile(0.95)
    df_feat['R_HighAmount'] = (df_feat['amount'] > threshold_amt).astype(int)
    
    # 2. Binary Rule: Transaction Frequency
    # "frequency of transactions" [cite: 408]
    freq = df_feat.groupby(['customer', 'step']).size().reset_index(name='count')
    df_feat = pd.merge(df_feat, freq, on=['customer', 'step'], how='left')
    df_feat['R_HighFreq'] = (df_feat['count'] > 3).astype(int) 
    df_feat.drop('count', axis=1, inplace=True)

    # 3. Binary Rule: Location (Zip Code Difference)
    # "criteria related to... location" 
    # BankSim has 'customer_zip' and 'merchant_zip'.
    # If they are different, it might be a risk (simplified rule)
    df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

    # 4. Binary Rule: Age Risk
    # Based on "Age Category" 
    df_feat['R_Age_Under25'] = (df_feat['age'] < 25).astype(int)
    df_feat['R_Age_Over65'] = (df_feat['age'] > 65).astype(int)
    
    # Features for the model (Omega set)
    # Paper uses "rules and properties... to create variable features" [cite: 311]
    numeric_cols = ['step', 'age', 'amount', 'R_HighAmount', 'R_HighFreq', 'R_DiffZip', 'R_Age_Under25', 'R_Age_Over65']
    
    scaler = MinMaxScaler()
    df_feat[numeric_cols] = scaler.fit_transform(df_feat[numeric_cols])
    
    return df_feat[numeric_cols], df_feat['fraud']

X_omega, y = create_paper_features(df_clean)
print(f"Feature Set (Ω) Shape: {X_omega.shape}")

# =============================================================================
# 3. DATA SPLITTING (PAPER SECTION 7.2.2)
# =============================================================================
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# 70% Normal for AE Training [cite: 417]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)

# Construct Test Set (50% of remaining Normals + All Fraud)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5))), np.ones(len(X_fraud))])

# Remaining Normals for EIF Training
X_train_eif = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"\nData Splits:")
print(f"1. AE Training: {X_train_ae.shape[0]}")
print(f"2. EIF Training: {X_train_eif.shape[0]}")
print(f"3. Testing: {X_test.shape[0]}")

# =============================================================================
# 4. AUTOENCODER TRAINING
# =============================================================================
# Using 14-7-14 Architecture for Dataset 2 [cite: 420]
# "trio of hidden layers of 14, 7, and 14 nodes"

print("\nStarting AutoEncoder training...")
input_dim = X_train_ae.shape[1]
encoding_dim1 = 14
encoding_dim2 = 7 # Bottleneck

input_layer = Input(shape=(input_dim,))
encoder1 = Dense(encoding_dim1, activation='relu')(input_layer)
bottleneck = Dense(encoding_dim2, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(encoding_dim1, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoder1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck) 

autoencoder.compile(optimizer='adam', loss='mse')

# Training with 110 Epochs as per paper 
history = autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
    verbose=1
)

# =============================================================================
# 5. GENERATING μ FEATURES
# =============================================================================
def extract_mu_features(data, ae_model, enc_model):
    if hasattr(data, 'values'):
        data_np = data.values
    else:
        data_np = data

    # A: Compressed Representation [cite: 342]
    A = enc_model.predict(data_np, verbose=0)
    
    # Reconstructions
    reconstructions = ae_model.predict(data_np, verbose=0)
    
    # B: Euclidean Distance [cite: 343]
    B = np.linalg.norm(data_np - reconstructions, axis=1).reshape(-1, 1)
    
    # C: Cosine Similarity [cite: 344]
    norm_data = np.linalg.norm(data_np, axis=1)
    norm_rec = np.linalg.norm(reconstructions, axis=1)
    dot_prod = np.sum(data_np * reconstructions, axis=1)
    C = (dot_prod / (norm_data * norm_rec + 1e-10)).reshape(-1, 1)
    
    return np.hstack([A, B, C]), reconstructions

print("\nExtracting μ features for EIF Training...")
mu_train, _ = extract_mu_features(X_train_eif, autoencoder, encoder_model)
X_train_eif_combined = np.hstack([X_train_eif.values, mu_train])

# =============================================================================
# 6. THRESHOLD DETERMINATION (OPTIMIZED)
# =============================================================================
print("\nDetermining Threshold...")
_, rec_train = extract_mu_features(X_train_ae.iloc[:5000], autoencoder, encoder_model)
mse_train = np.mean(np.power(X_train_ae.iloc[:5000].values - rec_train, 2), axis=1)

# Optimization: Using 80th percentile to increase Recall at AE stage
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE_THRESHOLD) 
print(f"Suspicion Threshold (AE MSE {AE_PERCENTILE_THRESHOLD}%): {THRESHOLD:.6f}")

# =============================================================================
# 7. EXTENDED ISOLATION FOREST TRAINING
# =============================================================================
print(f"\nTraining ExIFFI on {X_train_eif_combined.shape[0]} samples...")
# [cite: 422] "forest of 200 trees and a tree depth of 18"
eif_model = ExtendedIsolationForest(
    plus=True, 
    n_estimators=EIF_NTREES,
    max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]),
    max_depth=EIF_DEPTH
)
eif_model.fit(X_train_eif_combined)
print("EIF Training Completed.")

# =============================================================================
# 8. HYBRID DETECTION PIPELINE
# =============================================================================
print("\nRunning Hybrid Detection on Test Set...")

X_test_np = X_test.values
mu_test, rec_test = extract_mu_features(X_test_np, autoencoder, encoder_model)
mse_test = np.mean(np.power(X_test_np - rec_test, 2), axis=1)

final_preds = np.zeros(len(X_test_np))

# Step E: Contrast with Threshold [cite: 320]
suspicious_mask = mse_test > THRESHOLD
ae_flagged_count = np.sum(suspicious_mask)

if ae_flagged_count > 0:
    print(f"Processing {ae_flagged_count} suspicious samples with EIF...")
    X_suspicious = np.hstack([X_test_np[suspicious_mask], mu_test[suspicious_mask]])
    eif_scores = eif_model.predict(X_suspicious)
    final_preds[suspicious_mask] = eif_scores

# Step G: Final Decision [cite: 327]
final_binary_preds = (final_preds > 0.5).astype(int)

# =============================================================================
# 9. EVALUATION
# =============================================================================
print("\n" + "="*60)
print("EXPERIMENTAL RESULTS [cite: 467]")
print("="*60)
print(f"Suspects sent to EIF: {ae_flagged_count} ({ae_flagged_count/len(y_test)*100:.2f}%)")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_binary_preds))
print("\nClassification Report:")
print(classification_report(y_test, final_binary_preds))

acc = (confusion_matrix(y_test, final_binary_preds)[0,0] + confusion_matrix(y_test, final_binary_preds)[1,1]) / len(y_test)
print(f"Accuracy:  {acc:.4f} [cite: 432]")
print(f"F1-Score:  {f1_score(y_test, final_binary_preds):.4f} [cite: 448]")
print(f"MCC:       {matthews_corrcoef(y_test, final_binary_preds):.4f} [cite: 457]")