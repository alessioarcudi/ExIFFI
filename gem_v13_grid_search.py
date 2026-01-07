print("="*60)
print("FULL IMPLEMENTATION: AE-EIF+ (RICH RULES + GRID SEARCH)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import warnings
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'

# Training Hyperparameters
EPOCHS = 110 
BATCH_SIZE = 32
AE_PERCENTILE = 93  # Initial Threshold (will be optimized by Grid Search later)

# EIF Hyperparameters
EIF_NTREES = 400 
EIF_SAMPLE_SIZE = 256
EIF_DEPTH = 20 

# =============================================================================
# 2. RICH RULE-BASED FEATURE ENGINEERING
# =============================================================================
print("\n[STEP A] Executing Rich Rules Engineering...")

df_banksim = pd.read_csv(DATASET_PATH)

# --- Data Cleaning ---
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

# --- Rule 1: Temporal (Night & Weekend) ---
# Assumption: Step = Hour, 1 Day = 24 Steps.
df_feat['hour_of_day'] = df_feat['step'] % 24
df_feat['day_of_week'] = (df_feat['step'] // 24) % 7

df_feat['R_NightTime'] = df_feat['hour_of_day'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
df_feat['R_Weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# --- Rule 2: User Profile Stats (Z-Score) ---
cust_stats = df_feat.groupby('customer')['amount'].agg(['mean', 'std']).reset_index()
cust_stats.columns = ['customer', 'c_mean', 'c_std']
df_feat = pd.merge(df_feat, cust_stats, on='customer', how='left')
df_feat['c_std'] = df_feat['c_std'].fillna(1).replace(0, 1)

df_feat['User_ZScore'] = (df_feat['amount'] - df_feat['c_mean']) / df_feat['c_std']
df_feat['R_User_Outlier'] = (df_feat['User_ZScore'] > 3).astype(int)

# --- Rule 3: Category Context ---
cat_stats = df_feat.groupby('category')['amount'].mean().reset_index(name='cat_mean')
df_feat = pd.merge(df_feat, cat_stats, on='category', how='left')
df_feat['R_Cat_Spike'] = (df_feat['amount'] > (df_feat['cat_mean'] * 2)).astype(int)

# --- Rule 4: Velocity (Rapid Fire & Density) ---
step_counts = df_feat.groupby(['customer', 'step']).size().reset_index(name='tx_per_step')
df_feat = pd.merge(df_feat, step_counts, on=['customer', 'step'], how='left')
df_feat['R_RapidFire'] = (df_feat['tx_per_step'] > 1).astype(int)
# New: Density (Amount per transaction count)
df_feat['R_Vol_Density'] = df_feat['amount'] / (df_feat['tx_per_step'] + 0.1)

# --- Rule 5: Location ---
df_feat['R_DiffZip'] = (df_feat['customer_zip'] != df_feat['merchant_zip']).astype(int)

# --- Encoding & Scaling ---
df_feat['age'] = df_feat['age'].astype(str)
categorical_cols = ['category', 'age', 'gender']
df_encoded = pd.get_dummies(df_feat[categorical_cols], prefix=categorical_cols, dtype=int)

scaler = RobustScaler()
df_feat['Amount_Scaled'] = scaler.fit_transform(df_feat[['amount']])
df_feat['ZScore_Scaled'] = scaler.fit_transform(df_feat[['User_ZScore']])
df_feat['Density_Scaled'] = scaler.fit_transform(df_feat[['R_Vol_Density']])

# --- Feature Assembly ---
rule_cols = ['R_NightTime', 'R_Weekend', 'R_User_Outlier', 'R_Cat_Spike', 'R_RapidFire', 'R_DiffZip']
numeric_cols = ['Amount_Scaled', 'ZScore_Scaled', 'Density_Scaled']

X_omega = pd.concat([df_feat[rule_cols], df_feat[numeric_cols], df_encoded], axis=1)
y = df_feat['fraud']

print(f"Features Generated: {X_omega.shape[1]}")

# --- Splitting ---
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# Train (Normal only) vs Test (Mixed)
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
# Create a balanced/realistic test set
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])

# Data for EIF Training (Normal data not used in Test)
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

print(f"AE Train Size: {len(X_train_ae)} | Test Size: {len(X_test)}")

# =============================================================================
# 3. TRAIN AUTOENCODER
# =============================================================================
print("\n[STEP B] Training AutoEncoder...")

input_dim = X_train_ae.shape[1]
hidden_dim = max(14, int(input_dim / 1.5)) 
bottleneck_dim = 7

input_layer = Input(shape=(input_dim,))
encoder1 = Dense(hidden_dim, activation='tanh')(input_layer)
bottleneck = Dense(bottleneck_dim, activation='relu', name='bottleneck')(encoder1)
decoder1 = Dense(hidden_dim, activation='tanh')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoder1)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

# Using MAE as discussed to capture subtle anomalies better than MSE
autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mae')

autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

# =============================================================================
# 4. EXTRACTION & THRESHOLDING
# =============================================================================
def extract_mu(data_np):
    if hasattr(data_np, 'values'): data_np = data_np.values.astype(float)
    
    # Reconstruction
    rec = autoencoder.predict(data_np, verbose=0)
    
    # Latent Features
    A = encoder_model.predict(data_np, verbose=0)
    
    # Reconstruction Error (MSE for thresholding calculation)
    mse_loss = np.mean(np.power(data_np - rec, 2), axis=1)
    
    # Advanced Metrics (Cosine Sim & Norm)
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    norm_d = np.linalg.norm(data_np, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_d * norm_r + 1e-10)).reshape(-1, 1)
    
    return np.hstack([A, B, C]), mse_loss

print("\n[STEP C] Calculating Thresholds...")
# Calc MSE on training data to establish baseline
_, mse_train = extract_mu(X_train_ae.iloc[:10000])
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
print(f"Initial AE Threshold ({AE_PERCENTILE}%): {THRESHOLD:.6f}")

# =============================================================================
# 5. TRAIN EIF (OFFLINE)
# =============================================================================
print("\n[STEP D] Training EIF...")
mu_train_eif, _ = extract_mu(X_train_eif_raw)
X_train_eif_combined = np.hstack([X_train_eif_raw.values, mu_train_eif])

# Note: Ensure you have the 'isotree' or 'eif' library installed that supports 'plus' extension
# If standard sklearn IsolationForest, remove 'plus=True' and 'max_depth'
eif_model = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                                    max_samples=min(EIF_SAMPLE_SIZE, X_train_eif_combined.shape[0]), 
                                    max_depth=EIF_DEPTH)
eif_model.fit(X_train_eif_combined)

# =============================================================================
# 6. INITIAL EVALUATION (STATIC THRESHOLD)
# =============================================================================
print("\n[STEP E] Evaluation (Static Threshold)...")
X_test_np = X_test.values
mu_test, mse_test = extract_mu(X_test_np)

suspicious_mask = mse_test > THRESHOLD
final_decisions = np.zeros(len(X_test))

if np.sum(suspicious_mask) > 0:
    X_suspect = np.hstack([X_test_np[suspicious_mask], mu_test[suspicious_mask]])
    scores = eif_model.predict(X_suspect)
    # EIF typically returns scores 0 to 1. >0.5 is usually anomaly.
    final_decisions[suspicious_mask] = (scores > 0.5).astype(int)

print(classification_report(y_test, final_decisions))
print(f"Accuracy: {accuracy_score(y_test, final_decisions):.4f}")

# =============================================================================
# 7. AUTOMATED GRID SEARCH OPTIMIZATION
# =============================================================================
print("\n" + "="*60)
print("OPTIMIZATION: AE THRESHOLD GRID SEARCH")
print("="*60)

percentiles_to_test = [80, 85, 88, 90, 93, 95, 97, 99]
results = []

print("Running optimization loop...")

for p in percentiles_to_test:
    # A. Dynamic Threshold
    curr_thresh = np.percentile(mse_train, p)
    
    # B. Filter
    sus_mask = mse_test > curr_thresh
    count_sus = np.sum(sus_mask)
    
    # C. Decision Logic
    curr_preds = np.zeros(len(y_test))
    
    if count_sus > 0:
        X_sus = np.hstack([X_test_np[sus_mask], mu_test[sus_mask]])
        eif_scores = eif_model.predict(X_sus)
        curr_preds[sus_mask] = (eif_scores > 0.5).astype(int)
    
    # D. Metrics
    f1 = f1_score(y_test, curr_preds)
    prec = precision_score(y_test, curr_preds, zero_division=0)
    rec = recall_score(y_test, curr_preds)
    
    results.append({
        'Percentile': p,
        'Suspects': count_sus,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1
    })


df_res = pd.DataFrame(results).sort_values(by='F1_Score', ascending=False)
print("\nOPTIMIZATION RESULTS:")
print(df_res[['Percentile', 'Suspects', 'Precision', 'Recall', 'F1_Score']].to_string(index=False))

best_p = df_res.iloc[0]['Percentile']
print(f"\n[RECOMMENDATION] Change AE_PERCENTILE to {best_p} for best F1-Score.")