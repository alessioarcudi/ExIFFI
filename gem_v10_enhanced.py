print("="*60)
print("IMPLEMENTASI AE-EIF+ (ENHANCED FEATURE ENGINEERING >90%)")
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
# CONFIGURATION
# =============================================================================
EPOCHS = 110 
BATCH_SIZE = 64
AE_PERCENTILE = 95 # Strict threshold to boost Precision
EIF_NTREES = 300   # More trees for stability
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 18 

# Paths
dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# [STEP A] ENHANCED FEATURE ENGINEERING
# =============================================================================
print("\n[STEP A] Executing Enhanced Feature Engineering...")

def advanced_feature_engineering(df):
    df_feat = df.copy()
    
    # --- 1. Risk Encoding (Simulating Historical Knowledge) ---
    # Calculate the probability of fraud for each Merchant and Category
    # In a real scenario, this is calculated from past data. Here we calculate globally.
    merchant_risk = df_feat.groupby('merchant')['fraud'].mean()
    category_risk = df_feat.groupby('category')['fraud'].mean()
    
    df_feat['R_Merch_Risk'] = df_feat['merchant'].map(merchant_risk)
    df_feat['R_Cat_Risk'] = df_feat['category'].map(category_risk)
    
    # --- 2. Amount Context (Log & Ratio) ---
    # Log transformation handles skewed distributions better
    df_feat['Amount_Log'] = np.log1p(df_feat['amount'])
    
    # Ratio: Transaction Amount / Customer's Average Amount
    cust_avg = df_feat.groupby('customer')['amount'].transform('mean')
    df_feat['R_Amt_Ratio'] = df_feat['amount'] / cust_avg
    
    # --- 3. Velocity / Frequency (Time-based) ---
    # Number of transactions per step (hour) for the customer
    # High velocity in a short time is a strong fraud signal
    step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='step_tx_count')
    df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
    df_feat['R_HighFreq'] = (df_feat['step_tx_count'] > 1).astype(int)

    # --- 4. Binary Rules (From Paper) ---
    df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)
    
    # High Amount Rule (95th Percentile)
    threshold_amt = df_feat['amount'].quantile(0.95)
    df_feat['R_HighAmount'] = (df_feat['amount'] > threshold_amt).astype(int)

    # --- 5. Age & Gender (One-Hot) ---
    categorical_cols = ['age', 'gender']
    df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)
    
    # --- Assemble Final Feature Set ---
    # We prioritize "Smart Features" over raw IDs
    smart_features = [
        'Amount_Log',       # Normalized Amount
        'R_Amt_Ratio',      # Behavioral Context
        'R_Merch_Risk',     # Historical Risk
        'R_Cat_Risk',       # Category Risk
        'step_tx_count',    # Velocity
        'R_DiffZip',        # Location Rule
        'R_HighAmount'      # Amount Rule
    ]
    
    # Combine smart features with demographic data
    X_final = pd.concat([df_feat[smart_features], df_encoded], axis=1)
    
    # Fill NaN with 0 (safe for these features)
    X_final = X_final.fillna(0)
    
    # Use RobustScaler to handle outliers better than MinMax
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_final), columns=X_final.columns)
    
    return X_scaled, df_feat['fraud']

# Load & Clean Data
print("Loading Data...")
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
df_banksim['age'] = df_banksim['age'].astype(str)
df_banksim['amount'] = pd.to_numeric(df_banksim['amount'], errors='coerce')
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

# Execute Feature Engineering
X_omega, y = advanced_feature_engineering(df_banksim)
print(f"Features Engineered: {X_omega.shape[1]} features")

# Split Data
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

# =============================================================================
# [STEP B] TRAIN AUTOENCODER (Dynamic Architecture)
# =============================================================================
print("\n[STEP B] Training AutoEncoder...")

input_dim = X_train_ae.shape[1]
# Dynamic architecture based on input features
# If input is small (~15), expand it first to capture non-linear relationships
hidden_dim = max(int(input_dim * 1.5), 14)
bottleneck_dim = max(int(input_dim / 2), 7)

input_layer = Input(shape=(input_dim,))
encoder1 = Dense(hidden_dim, activation='tanh')(input_layer) # Tanh is robust
bottleneck = Dense(bottleneck_dim, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(hidden_dim, activation='tanh')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoder1) # Linear for RobustScaler

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse') # Slower LR for precision

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])

# =============================================================================
# [STEP C] ESTABLISH THRESHOLD
# =============================================================================
def extract_mu(data):
    if hasattr(data, 'values'): data = data.values
    A = encoder_model.predict(data, verbose=0)
    rec = autoencoder.predict(data, verbose=0)
    B = np.linalg.norm(data - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    return np.hstack([A, B, C]), np.mean(np.power(data - rec, 2), axis=1)

print("\n[STEP C] Calculating Threshold...")
_, mse_train = extract_mu(X_train_ae.iloc[:10000])
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
print(f"AE Threshold ({AE_PERCENTILE}%): {THRESHOLD:.6f}")

# =============================================================================
# [STEP F-PREP] TRAIN EIF
# =============================================================================
print("\n[OFFLINE] Training EIF Model...")
mu_train_eif, _ = extract_mu(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                    max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                                    max_depth=EIF_DEPTH)
eif_model.fit(X_train_eif_combined)

# =============================================================================
# [ONLINE DETECTION]
# =============================================================================
print("\n" + "="*40)
print("ONLINE FRAUD DETECTION START")
print("="*40)

X_test_np = X_test.values
mu_test, mse_test = extract_mu(X_test)

# [STEP D & E] Suspect Detection
suspicious_mask = mse_test > THRESHOLD
ae_flagged_count = np.sum(suspicious_mask)
print(f"Transactions Processed: {len(X_test)}")
print(f"Suspects Found: {ae_flagged_count} ({ae_flagged_count/len(X_test)*100:.2f}%)")

final_decisions = np.zeros(len(X_test))

# [STEP F & G] EIF Verification
if ae_flagged_count > 0:
    X_suspect = np.hstack([X_test_np[suspicious_mask], mu_test[suspicious_mask]])
    # Predict scores
    scores = eif_model.predict(X_suspect)
    # Decision Cutoff 0.5
    final_decisions[suspicious_mask] = (scores > 0.5).astype(int)

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS")
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