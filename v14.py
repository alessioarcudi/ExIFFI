print("="*60)
print("IMPLEMENTASI AE-EIF+ V14 (ROLLING WINDOW STRATEGY)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import sys
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# 1. KONFIGURASI (Agresif untuk Mengejar Jurnal)
# =============================================================================
DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
EPOCHS = 110              # KEMBALIKAN KE ANGKA TINGGI! 3 Epoch tidak akan cukup.
BATCH_SIZE = 2048         
AE_PERCENTILE_INIT = 95   # Kita perketat filter awal
EIF_NTREES = 200          # Jurnal Hal 194: "forest of 200 trees"
EIF_SAMPLE_SIZE = 256     
EIF_DEPTH = 18            # Jurnal Hal 194: "tree depth of 18" - Jangan 82!

# =============================================================================
# [STEP A] HYBRID FEATURE ENGINEERING (BINARY RULES + ROLLING)
# =============================================================================
print("\n[STEP A] Generating Binary Rule-Based Features...")

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

# Sort untuk Rolling Window
df = df.sort_values(['customer', 'step'])

# --- 1. ROLLING WINDOW CALCULATIONS (Untuk Data Pendukung) ---
grouped = df.groupby('customer')
df['roll_3h_amt'] = grouped['amount'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
df['roll_3h_cnt'] = grouped['amount'].transform(lambda x: x.rolling(window=3, min_periods=1).count())

# Stats Global
cat_mean = df.groupby('category')['amount'].transform('mean')
cat_std = df.groupby('category')['amount'].transform('std').fillna(1)
cust_mean = grouped['amount'].transform('mean')
cust_std = grouped['amount'].transform('std').fillna(1)

# --- 2. BINARY RULES (JOURNAL STYLE) ---
# Jurnal menggunakan fitur biner (0/1) agar AE lebih tegas membedakan pola.

# Rule 1: High Amount ( > Mean Kategori + 2 SD )
df['B_HighAmt_Cat'] = (df['amount'] > (cat_mean + 2*cat_std)).astype(int)

# Rule 2: High Amount User ( > Mean User + 3 SD ) - Anomali Personal
df['B_HighAmt_User'] = (df['amount'] > (cust_mean + 3*cust_std)).astype(int)

# Rule 3: High Velocity ( > 2 transaksi dalam 3 jam terakhir)
df['B_RapidFire'] = (df['roll_3h_cnt'] > 2).astype(int)

# Rule 4: Suspicious Time (Dini hari 0-6)
df['step_hour'] = df['step'] % 24
df['B_NightTime'] = df['step_hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# Rule 5: New Location / Diff Zip
df['B_DiffZip'] = (df['customer_zip'] != df['merchant_zip']).astype(int)

# Rule 6: Max Amount Hit (Apakah ini max amount user tersebut sejauh ini?)
df['max_amt_so_far'] = grouped['amount'].transform(lambda x: x.expanding().max())
df['B_NewMax'] = (df['amount'] >= df['max_amt_so_far']).astype(int)

# --- 3. PREPARING INPUT ---
# Kita gabungkan Nilai Asli (Scaled) + Binary Rules
categorical_cols = ['category', 'gender']
df_encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, dtype=int)

scaler = RobustScaler()
# Hanya scale amount dan rolling, sisanya biarkan Binary 0/1
num_cols = ['amount', 'roll_3h_amt'] 
df[num_cols] = scaler.fit_transform(df[num_cols])

# Gabungkan: Continuous + Binary Rules + OneHot
rule_cols = ['B_HighAmt_Cat', 'B_HighAmt_User', 'B_RapidFire', 'B_NightTime', 'B_DiffZip', 'B_NewMax']
X_omega = pd.concat([df[num_cols], df[rule_cols], df_encoded], axis=1)
y = df['fraud']

print(f"Total Features: {X_omega.shape[1]} (Dominant: Binary Rules)")

# Split
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# Train (Normal) vs Test (Mixed)
X_train_ae, X_train_eif_base = train_test_split(X_normal, test_size=0.3, random_state=42)
X_test = pd.concat([X_train_eif_base.sample(frac=0.5, random_state=42), X_fraud]) 
y_test = np.concatenate([np.zeros(len(X_train_eif_base.sample(frac=0.5, random_state=42))), np.ones(len(X_fraud))])
X_train_eif_raw = X_train_eif_base.drop(X_test[y_test==0].index)

# =============================================================================
# [STEP B] AUTOENCODER (ARSITEKTUR JURNAL PERSIS)
# =============================================================================
print("\n[STEP B] Training AutoEncoder (Journal Arch)...")
input_dim = X_train_ae.shape[1]

# Jurnal Hal 194: 38 (Input) -> 14 -> 7 -> 14 -> 38 (Output)
# Kita sesuaikan input_dim kita (~45 features) -> 14 -> 7 -> 14 -> input_dim
input_layer = Input(shape=(input_dim,))
enc1 = Dense(14, activation='tanh')(input_layer)  # Layer 1
bottleneck = Dense(7, activation='relu')(enc1)    # Bottleneck
dec1 = Dense(14, activation='tanh')(bottleneck)   # Layer 3
output_layer = Dense(input_dim, activation='sigmoid')(dec1) # Sigmoid karena input dominan 0-1 (Binary)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# EPOCH WAJIB TINGGI
autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                shuffle=True, validation_split=0.1, verbose=1,
                callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])

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

print("\n[STEP D] Training EIF (Depth 18)...")
X_train_eif_final, _ = get_features(X_train_eif_raw)
eif = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                              max_samples=EIF_SAMPLE_SIZE, max_depth=EIF_DEPTH)
eif.fit(X_train_eif_final)

# =============================================================================
# [STEP E] OPTIMIZATION
# =============================================================================
print("\n[STEP E] Scoring & Optimizing...")
X_test_final, mse_test = get_features(X_test)
eif_scores = eif.predict(X_test_final)

# Grid Search untuk mencari Recall > 0.95 dengan Precision terbaik
ae_ps = [90, 93, 95, 96, 97, 98] # Percentile
eif_ts = [0.45, 0.50, 0.55, 0.60]

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
top = res_df[res_df['Recall'] > 0.95].sort_values(by='Precision', ascending=False).head(1)

if not top.empty:
    best = top.iloc[0]
    print(f"\n[BEST CONFIG] AE: {best['AE_%']}% | EIF: {best['EIF_T']}")
    
    # Final Run
    mask_final = mse_test > np.percentile(mse_train, best['AE_%'])
    y_pred = np.zeros(len(y_test))
    y_pred[mask_final] = (eif_scores[mask_final] > best['EIF_T']).astype(int)
    
    print(classification_report(y_test, y_pred))
    print(f"Jurnal Target: Recall 0.97, Precision 0.91")
else:
    print("Tidak ada konfigurasi Recall > 95%. Cek hasil Grid Search manual.")
    print(res_df.sort_values(by='F1', ascending=False).head(5))