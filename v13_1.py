print("="*60)
print("IMPLEMENTASI AE-EIF+ (DATASET 1: CREDIT CARD)")
print("BASED ON: Abbassi et al. (2024) Methodology")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, matthews_corrcoef
import warnings

# Import library EIF Anda (pastikan mendukung parameter plus=True)
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *

# --- CONFIG ---
dataset_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/creditcard.csv'
ENSEMBLE_ROUNDS = 5     
EIF_NTREES = 300        
EIF_SAMPLE_SIZE = 512   
TOP_K = 550             

# --- 1. PREPROCESSING ---
print("\n[STEP 1] Loading & Preprocessing...")
df = pd.read_csv(dataset_path)

rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
# df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1)) # Time tidak muncul di top features Anda
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# --- REVISI KRUSIAL DISINI ---
# Menggunakan Top-12 fitur persis dari grafik output.png Anda
# Urutan: V23, V21, V28, V27, V20, V8, Amount, V2, V10, V7, V14, V17
IMPORTANT_FEATURES = [
    'V23', 'V21', 'V28', 'V27', 'V20', 'V8', 
    'scaled_amount', 'V2', 'V10', 'V7', 'V14', 'V17'
]
print(f"Using Data-Driven Top Features: {IMPORTANT_FEATURES}")

X_omega = df.drop('Class', axis=1)
y = df['Class']

# Split Data 
X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

X_train_ae, X_rem = train_test_split(X_normal, train_size=0.7, random_state=42)
X_train_eif_base, X_test_normal = train_test_split(X_rem, test_size=0.5, random_state=42)

X_test = pd.concat([X_test_normal, X_fraud])
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

# --- 2. AUTOENCODER (Tetap Gunakan Semua Fitur untuk Global Context) ---
print("\n[STEP 2] Training AutoEncoder...")
input_dim = X_train_ae.shape[1]
input_layer = Input(shape=(input_dim,))
enc = Dense(14, activation='relu')(input_layer)
bottleneck = Dense(7, activation='relu')(enc)
dec = Dense(14, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(dec)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Training singkat tapi efektif
autoencoder.fit(X_train_ae, X_train_ae, epochs=5, batch_size=2048, 
                shuffle=True, validation_split=0.1, verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

def get_hybrid_input(data_df):
    # 1. Fitur AE (Latent + Error + Similarity)
    data_np = data_df.values
    A = encoder_model.predict(data_np, verbose=0)
    rec = autoencoder.predict(data_np, verbose=0)
    B = np.linalg.norm(data_np - rec, axis=1).reshape(-1, 1)
    
    norm_in = np.linalg.norm(data_np, axis=1)
    norm_rec = np.linalg.norm(rec, axis=1)
    C = (np.sum(data_np * rec, axis=1) / (norm_in * norm_rec + 1e-10)).reshape(-1, 1)
    
    mu_features = np.hstack([A, B, C])
    
    # 2. Fitur Selektif (Hanya yang muncul di grafik Anda)
    selected_data = data_df[IMPORTANT_FEATURES].values
    
    # Gabung
    return np.hstack([selected_data, mu_features]), B.flatten()

# --- 3. ENSEMBLE EIF TRAINING ---
print(f"\n[STEP 3] Training Ensemble EIF+ on Selected Features...")

X_train_eif_final, _ = get_hybrid_input(X_train_eif_base)

eif_models = []
for i in range(ENSEMBLE_ROUNDS):
    # Variasi random state jika library mendukung, atau biarkan fit melakukan random sampling
    print(f"  > Model {i+1}/{ENSEMBLE_ROUNDS}...")
    model = ExtendedIsolationForest(plus=True, 
                                    n_estimators=EIF_NTREES, 
                                    max_samples=EIF_SAMPLE_SIZE, 
                                    max_depth=18)
    model.fit(X_train_eif_final)
    eif_models.append(model)

# --- 4. PREDICTION ---
print("\n[STEP 4] Prediction & Scoring...")
X_test_final, mse_test = get_hybrid_input(X_test)

# Filter AE
ae_threshold = np.percentile(mse_test, 90) 
suspicious_mask = mse_test > ae_threshold
print(f"  > AE Pre-filter candidates: {np.sum(suspicious_mask)}")

aggregated_scores = np.zeros(len(X_test))

if np.sum(suspicious_mask) > 0:
    X_suspect = X_test_final[suspicious_mask]
    total_score = np.zeros(len(X_suspect))
    
    for model in eif_models:
        total_score += model.predict(X_suspect)
        
    aggregated_scores[suspicious_mask] = total_score / ENSEMBLE_ROUNDS

# --- FINAL TOP-K ---
print(f"\n[FINAL] Top-{TOP_K} Cutoff...")
sorted_indices = np.argsort(aggregated_scores)[::-1]
top_k_indices = sorted_indices[:TOP_K]

final_preds = np.zeros(len(X_test))
final_preds[top_k_indices] = 1

# --- EVALUASI ---
cm = confusion_matrix(y_test, final_preds)
tp = cm[1, 1]
fp = cm[0, 1]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / np.sum(y_test == 1)

print("\n" + "="*60)
print(f"FINAL RESULT (Data-Driven Features)")
print("="*60)
print(cm)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {2 * (precision * recall) / (precision + recall):.4f}")