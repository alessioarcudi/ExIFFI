print("="*60)
print("DYNAMIC TUNING: AE-EIF+ (RICH BEHAVIORAL RULES)")
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
import itertools

warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')

# =============================================================================
# 1. PARAMETER GRID (Search Space)
# =============================================================================
# Define the parameters you want to tune
param_grid = {
    'batch_size': [1024, 2048],
    'epochs': [30, 50],
    'ae_percentile': [85, 90, 95],
    'eif_ntrees': [200, 300],
    'eif_sample_size': [256, 512],
    'eif_depth': [15, 20]
}

dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# =============================================================================
# [STEP A] DATA PREPARATION (Run Once)
# =============================================================================
print("\n[PREP] Preparing Data & Features...")

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
df_banksim['amount'] = pd.to_numeric(df_banksim['amount'], errors='coerce')
df_banksim['fraud'] = pd.to_numeric(df_banksim['fraud'], errors='coerce').fillna(0).astype(int)

df_feat = df_banksim.copy()

# Feature Engineering
df_feat['hour_of_day'] = df_feat['step'] % 24
df_feat['day_of_week'] = (df_feat['step'] // 24) % 7
df_feat['R_NightTime'] = df_feat['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
df_feat['R_Weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

cust_stats = df_feat.groupby('customer')['amount'].agg(['mean', 'std']).reset_index()
cust_stats.columns = ['customer', 'c_mean', 'c_std']
df_feat = pd.merge(df_feat, cust_stats, on='customer', how='left')
df_feat['c_std'] = df_feat['c_std'].fillna(1).replace(0, 1)

df_feat['User_ZScore'] = (df_feat['amount'] - df_feat['c_mean']) / df_feat['c_std']
df_feat['R_User_Outlier'] = (df_feat['User_ZScore'] > 3).astype(int)

cat_stats = df_feat.groupby('category')['amount'].mean().reset_index(name='cat_mean')
df_feat = pd.merge(df_feat, cat_stats, on='category', how='left')
df_feat['R_Cat_Spike'] = (df_feat['amount'] > (df_feat['cat_mean'] * 2)).astype(int)

df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

df_feat['age'] = df_feat['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

scaler = RobustScaler()
df_feat['Amount_Scaled'] = scaler.fit_transform(df_feat[['amount']])
df_feat['ZScore_Scaled'] = scaler.fit_transform(df_feat[['User_ZScore']])

rule_cols = ['R_NightTime', 'R_Weekend', 'R_User_Outlier', 'R_Cat_Spike', 'R_DiffZip']
numeric_cols = ['Amount_Scaled', 'ZScore_Scaled']

X_omega = pd.concat([df_feat[rule_cols], df_feat[numeric_cols], df_encoded], axis=1)
y = df_feat['fraud']

# Splitting Data
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# Split Train/Test once to ensure fair comparison
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"Data Prepared. AE Train: {len(X_train_ae)} | EIF Train: {len(X_train_eif_raw)} | Test: {len(y_test)}")

# =============================================================================
# [STEP B] GRID SEARCH FUNCTION
# =============================================================================
def run_experiment(params):
    print(f"\nTesting Params: {params}")
    
    # 1. Train AutoEncoder
    input_dim = X_train_ae.shape[1]
    hidden_dim = max(14, int(input_dim / 1.5)) 
    bottleneck_dim = 7

    input_layer = Input(shape=(input_dim,))
    encoder1 = Dense(hidden_dim, activation='tanh')(input_layer)
    bottleneck = Dense(bottleneck_dim, activation='relu')(encoder1)
    decoder1 = Dense(hidden_dim, activation='tanh')(bottleneck)
    output_layer = Dense(input_dim, activation='sigmoid')(decoder1)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder_model = Model(inputs=input_layer, outputs=bottleneck)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    # Train AE
    autoencoder.fit(X_train_ae, X_train_ae, 
                    epochs=params['epochs'], 
                    batch_size=params['batch_size'], 
                    shuffle=True, 
                    validation_split=0.1, 
                    verbose=0, # Silent mode
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    # 2. Extract Features
    def extract_mu(data_np):
        if hasattr(data_np, 'values'): data_np = data_np.values.astype(float)
        A = encoder_model.predict(data_np, verbose=0)
        rec = autoencoder.predict(data_np, verbose=0)
        mse_loss = np.mean(np.power(data_np - rec, 2), axis=1)
        
        norm_d = np.linalg.norm(data_np, axis=1)
        norm_r = np.linalg.norm(rec, axis=1)
        C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
        
        return np.hstack([A, mse_loss.reshape(-1,1), C]), mse_loss

    # 3. Calculate Threshold
    _, mse_train = extract_mu(X_train_ae.iloc[:10000])
    THRESHOLD = np.percentile(mse_train, params['ae_percentile'])

    # 4. Train EIF
    mu_train_eif, _ = extract_mu(X_train_eif_raw)
    X_train_eif_final = mu_train_eif # Using hybrid features directly
    
    eif_model = ExtendedIsolationForest(plus=True, 
                                        n_estimators=params['eif_ntrees'], 
                                        max_samples=min(params['eif_sample_size'], X_train_eif_final.shape[0]), 
                                        max_depth=params['eif_depth'])
    eif_model.fit(X_train_eif_final)

    # 5. Prediction
    X_test_np = X_test.values
    mu_test, mse_test = extract_mu(X_test_np)
    
    suspicious_mask = mse_test > THRESHOLD
    final_decisions = np.zeros(len(X_test))
    
    if np.sum(suspicious_mask) > 0:
        X_suspect = mu_test[suspicious_mask]
        scores = eif_model.predict(X_suspect)
        final_decisions[suspicious_mask] = (scores > 0.5).astype(int)

    # 6. Evaluation
    f1 = f1_score(y_test, final_decisions)
    acc = accuracy_score(y_test, final_decisions)
    return f1, acc

# =============================================================================
# [STEP C] RUNNING GRID SEARCH
# =============================================================================
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\nStarting Grid Search with {len(combinations)} combinations...")
best_score = 0
best_params = {}

results_log = []

for i, params in enumerate(combinations):
    start_time = time.time()
    try:
        f1, acc = run_experiment(params)
        duration = time.time() - start_time
        
        print(f"   Result [{i+1}/{len(combinations)}]: F1={f1:.4f} | Acc={acc:.4f} | Time={duration:.1f}s")
        
        results_log.append({**params, 'f1': f1, 'accuracy': acc, 'time': duration})
        
        if f1 > best_score:
            best_score = f1
            best_params = params
            print(f"   >>> NEW BEST FOUND! F1: {best_score:.4f}")
            
    except Exception as e:
        print(f"   Error with params {params}: {str(e)}")

# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "="*60)
print("GRID SEARCH COMPLETE")
print("="*60)
print(f"Best F1-Score: {best_score:.4f}")
print("Best Parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Save results to CSV for analysis
df_results = pd.DataFrame(results_log)
df_results = df_results.sort_values(by='f1', ascending=False)
print("\nTop 5 Configurations:")
print(df_results.head(5).to_string(index=False))

# Optional: You can save this to a CSV file
df_results.to_csv('tuning_results.csv', index=False)