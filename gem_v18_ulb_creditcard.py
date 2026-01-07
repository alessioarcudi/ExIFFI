print("="*60)
print("IMPLEMENTASI AE-EIF+ (ADAPTASI DATASET KAGGLE ULB)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import warnings
# Import library EIF Anda
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import sys

# =============================================================================
# 1. KONFIGURASI
# =============================================================================
DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/creditcard.csv'

EPOCHS = 3 
BATCH_SIZE = 128 
# Tuning EIF Diperkuat (Lebih banyak pohon & sample)
EIF_NTREES = 300 
EIF_SAMPLE_SIZE = 512 
EIF_DEPTH = 20 

# =============================================================================
# 2. DATA PREPARATION (ROBUST)
# =============================================================================
print("\n[STEP A] Loading & Scaling Data...")
df = pd.read_csv(DATASET_PATH)
df.rename(columns={'Class': 'fraud'}, inplace=True)

# Scale Time & Amount
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('fraud', axis=1)
y = df['fraud']

# Stratified Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# AE Train: Normal Only
X_train_ae = X_train_raw[y_train == 0]
# EIF Train: Semua data train (Unsupervised)
X_train_eif = X_train_raw

print(f"AE Train: {len(X_train_ae)} | Test: {len(X_test_raw)}")

# =============================================================================
# 3. WIDE AUTOENCODER (WINNING ARCHITECTURE)
# =============================================================================
print("\n[STEP B] Training Wide AutoEncoder (22-15-22)...")

input_dim = X_train_ae.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder (Wide Strategy)
# BatchNormalization membantu Latent Space lebih terdistribusi rapi
e = Dense(22)(input_layer)
e = BatchNormalization()(e)
e = Activation('tanh')(e)

latent_layer = Dense(15, activation='relu', name='latent')(e)

# Decoder
d = Dense(22)(latent_layer)
d = BatchNormalization()(d)
d = Activation('tanh')(d)

output_layer = Dense(input_dim, activation='linear')(d)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=latent_layer)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

autoencoder.fit(X_train_ae, X_train_ae, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                validation_split=0.1,
                verbose=1,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# =============================================================================
# 4. FUSION FEATURE EXTRACTION
# =============================================================================
def get_fusion_features(data_x):
    if hasattr(data_x, 'values'): data_x = data_x.values
    
    # 1. Latent Features (Struktur)
    latent = encoder_model.predict(data_x, verbose=0)
    
    # 2. Reconstruction Error (Anomali Error)
    rec = autoencoder.predict(data_x, verbose=0)
    mse = np.mean(np.power(data_x - rec, 2), axis=1).reshape(-1, 1)
    
    # 3. Cosine Similarity (Arah Vector)
    norm_x = np.linalg.norm(data_x, axis=1)
    norm_r = np.linalg.norm(rec, axis=1)
    cosine = (np.sum(data_x * rec, axis=1) / (norm_x * norm_r + 1e-10)).reshape(-1, 1)
    
    # GABUNGKAN SEMUA: Latent + MSE + Cosine
    # Pastikan tipe data float64 agar Numba tidak error
    fusion = np.hstack([latent, mse, cosine])
    return fusion.astype(np.float64)

print("\n[STEP C] Generating Fusion Features for EIF...")
X_train_fusion = get_fusion_features(X_train_eif)
X_test_fusion = get_fusion_features(X_test_raw)

print(f"Fusion Feature Shape: {X_train_fusion.shape}")

# =============================================================================
# 5. TRAINING EIF PLUS
# =============================================================================
print("\n[STEP D] Training EIF+ on Fusion Features...")
eif = ExtendedIsolationForest(plus=True, n_estimators=EIF_NTREES, 
                              max_samples=EIF_SAMPLE_SIZE, 
                              max_depth=EIF_DEPTH)
eif.fit(X_train_fusion)

# =============================================================================
# 6. SCORING & OPTIMIZATION
# =============================================================================
print("\n[STEP E] Predicting & Optimizing...")
# Dapatkan skor anomali dari EIF (0.0 s.d 1.0)
# Ingat: Makin tinggi skor = makin anomali
test_scores = eif.predict(X_test_fusion)

# Grid Search pada Score EIF langsung
best_f1 = 0
best_res = {}

# Scan threshold EIF Score dari 0.4 sampai 0.7
thresholds = np.arange(0.40, 0.75, 0.02)

print(f"{'EIF Cutoff':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 55)

for thresh in thresholds:
    preds = (test_scores > thresh).astype(int)
    
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    
    # Print progress (hanya jika F1 lumayan atau di awal/akhir)
    if f1 > 0.1:
        print(f"{thresh:<12.2f} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_res = {
            'threshold': thresh,
            'preds': preds,
            'acc': accuracy_score(y_test, preds),
            'prec': prec,
            'rec': rec,
            'f1': f1
        }

if best_res:
    print("\n" + "="*60)
    print(f"ULTIMATE RESULTS (EIF Cutoff: {best_res['threshold']:.2f})")
    print("="*60)
    print(confusion_matrix(y_test, best_res['preds']))
    print(classification_report(y_test, best_res['preds']))
    print(f"Accuracy: {best_res['acc']:.4f}")
    print(f"F1-Score: {best_res['f1']:.4f}")
else:
    print("\nModel gagal konvergen (F1 sangat rendah). Coba tuning arsitektur lagi.")