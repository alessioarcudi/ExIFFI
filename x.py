print("="*60)
print("OPTIMIZED AE-EIF+ FOR 90%+ PERFORMANCE (WITH EIF)")
print("="*60)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from utils_reboot.datasets import Dataset
from utils_reboot.utils import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from model_reboot import *
import warnings
import joblib


warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# 1. ULTRA-OPTIMIZED CONFIGURATION
# =============================================================================
DATASET_PATH = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
EPOCHS = 5               # More epochs for better convergence
BATCH_SIZE = 4096          # Larger batch size for stability
AE_PERCENTILE = 80       # Very strict AE threshold
EIF_NTREES = 100           # More trees for better precision
EIF_SAMPLE_SIZE = 128      # Balanced sample size
EIF_DEPTH = 14             # Deeper trees for complex patterns

# =============================================================================
# [STEP A] ADVANCED FEATURE ENGINEERING WITH FRAUD PATTERNS
# =============================================================================
print("\n[STEP A] Advanced Feature Engineering...")

df = pd.read_csv(DATASET_PATH)

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

# ========== 1. TEMPORAL PATTERNS WITH CYCLICAL ENCODING ==========
print("  1. Creating temporal patterns...")
df['hour'] = df['step'] % 24
df['day_of_week'] = (df['step'] // 24) % 7
df['day_of_month'] = (df['step'] // (24 * 7)) % 30

# Cyclical encoding for all temporal features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
df['month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)

# ========== 2. CUSTOMER BEHAVIOR PROFILING ==========
print("  2. Building customer behavior profiles...")
grouped = df.groupby('customer')

# Basic customer stats
df['customer_total_trans'] = grouped['amount'].transform('count')
df['customer_total_amount'] = grouped['amount'].transform('sum')
df['customer_avg_amount'] = grouped['amount'].transform('mean')
df['customer_std_amount'] = grouped['amount'].transform('std').fillna(0)
df['customer_max_amount'] = grouped['amount'].transform('max')
df['customer_min_amount'] = grouped['amount'].transform('min')
df['customer_median_amount'] = grouped['amount'].transform('median')

# Customer transaction frequency
df['customer_avg_time_gap'] = grouped['step'].diff().abs().groupby(df['customer']).transform('mean').fillna(24)

# ========== 3. ADVANCED ROLLING STATISTICS ==========
print("  3. Calculating advanced rolling statistics...")
windows = [1, 2, 3, 6, 12, 24, 48]
for w in windows:
    # Rolling sums
    df[f'roll_{w}h_sum'] = grouped['amount'].transform(lambda x: x.rolling(window=w, min_periods=1).sum())
    df[f'roll_{w}h_count'] = grouped['amount'].transform(lambda x: x.rolling(window=w, min_periods=1).count())
    df[f'roll_{w}h_mean'] = grouped['amount'].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
    df[f'roll_{w}h_std'] = grouped['amount'].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
    
    # Rolling percentiles
    if w >= 3:
        df[f'roll_{w}h_q25'] = grouped['amount'].transform(lambda x: x.rolling(window=w, min_periods=1).quantile(0.25))
        df[f'roll_{w}h_q75'] = grouped['amount'].transform(lambda x: x.rolling(window=w, min_periods=1).quantile(0.75))

# ========== 4. VELOCITY AND ACCELERATION FEATURES ==========
print("  4. Computing velocity and acceleration...")
df['time_since_last'] = grouped['step'].diff().fillna(24)
df['amount_per_hour'] = df['amount'] / (df['time_since_last'] + 0.1)

# Velocity ratios
df['velocity_1h_24h'] = df['roll_1h_sum'] / (df['roll_24h_sum'] + 1)
df['velocity_3h_24h'] = df['roll_3h_sum'] / (df['roll_24h_sum'] + 1)
df['velocity_1h_3h'] = df['roll_1h_sum'] / (df['roll_3h_sum'] + 1)

# Acceleration (change in velocity)
df['acceleration_1h'] = df['roll_1h_sum'] - df['roll_1h_sum'].shift(1).fillna(0)

# ========== 5. STATISTICAL ANOMALY DETECTORS ==========
print("  5. Building statistical anomaly detectors...")
# Z-scores at multiple levels
df['zscore_customer'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 0.01)
df['zscore_global'] = (df['amount'] - df['amount'].mean()) / (df['amount'].std() + 0.01)

# Category-based statistics
category_stats = df.groupby('category')['amount'].agg(['mean', 'std', 'count']).fillna(0)
df['category_mean'] = df['category'].map(category_stats['mean'])
df['category_std'] = df['category'].map(category_stats['std'])
df['category_count'] = df['category'].map(category_stats['count'])
df['zscore_category'] = (df['amount'] - df['category_mean']) / (df['category_std'] + 0.01)

# IQR-based anomaly scores
Q1 = df.groupby('customer')['amount'].transform(lambda x: x.quantile(0.25))
Q3 = df.groupby('customer')['amount'].transform(lambda x: x.quantile(0.75))
IQR = Q3 - Q1
df['iqr_score'] = (df['amount'] - Q3) / (IQR + 0.01)
df['iqr_ratio'] = df['amount'] / (Q3 + 1)

# ========== 6. MERCHANT RISK PROFILING ==========
print("  6. Creating merchant risk profiles...")
merchant_stats = df.groupby('merchant').agg({
    'fraud': ['mean', 'sum', 'count'],
    'amount': ['mean', 'std', 'max']
}).fillna(0)

merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns.values]

# Map merchant stats back to dataframe
for col in merchant_stats.columns:
    df[f'merch_{col}'] = df['merchant'].map(merchant_stats[col])

# Merchant risk scores
df['merch_fraud_rate'] = df['merch_fraud_mean']
df['merch_risk_score'] = df['merch_fraud_rate'] * np.log1p(df['merch_fraud_count'])
df['merch_amount_risk'] = df['merch_amount_std'] * df['merch_fraud_rate']

# ========== 7. CUSTOMER-MERCHANT INTERACTIONS ==========
print("  7. Analyzing customer-merchant interactions...")
# Customer-merchant history
cust_merch_stats = df.groupby(['customer', 'merchant']).agg({
    'amount': ['count', 'sum', 'mean', 'std'],
    'step': ['min', 'max']
}).fillna(0)

cust_merch_stats.columns = ['_'.join(col).strip() for col in cust_merch_stats.columns.values]

# Merge back to dataframe
for col in ['amount_count', 'amount_sum', 'amount_mean', 'amount_std']:
    df[f'cust_merch_{col}'] = df.set_index(['customer', 'merchant']).index.map(
        lambda idx: cust_merch_stats.loc[idx, col] if idx in cust_merch_stats.index else 0
    )

# New merchant flag
df['is_new_merchant'] = (df['cust_merch_amount_count'] <= 1).astype(int)

# Customer-merchant z-score
df['zscore_cust_merch'] = (df['amount'] - df['cust_merch_amount_mean']) / (df['cust_merch_amount_std'] + 0.01)

# ========== 8. CATEGORY-TIME INTERACTIONS ==========
print("  8. Creating category-time interactions...")
df['category_hour'] = df['category'].astype(str) + "_" + df['hour'].astype(str)
cat_hour_stats = df.groupby('category_hour')['amount'].agg(['mean', 'std', 'count']).fillna(0)

for stat in ['mean', 'std', 'count']:
    df[f'cat_hour_{stat}'] = df['category_hour'].map(cat_hour_stats[stat])

df['zscore_cat_hour'] = (df['amount'] - df['cat_hour_mean']) / (df['cat_hour_std'] + 0.01)

# ========== 9. GEOGRAPHICAL AND DEMOGRAPHIC FEATURES ==========
print("  9. Adding geographical features...")
df['same_zip'] = (df['customer_zip'] == df['merchant_zip']).astype(int)

# Zip code risk (frequency-based)
zip_freq = df['merchant_zip'].value_counts(normalize=True)
df['merchant_zip_freq'] = df['merchant_zip'].map(zip_freq).fillna(0)

# Age and gender encoding
df['age_numeric'] = pd.to_numeric(df['age'].str.replace('U', '0').str.replace('J', '1'), errors='coerce').fillna(0)
df['gender_numeric'] = df['gender'].map({'U': 0, 'F': 1, 'M': 2}).fillna(0)

# ========== 10. BINARY FRAUD INDICATORS ==========
print("  10. Creating binary fraud indicators...")
# Time-based indicators
df['R_Night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
df['R_Weekend'] = (df['day_of_week'] >= 5).astype(int)
df['R_Rapid'] = (df['time_since_last'] < 0.5).astype(int)  # Less than 30 minutes

# Amount-based indicators
df['R_HighZScore'] = (df['zscore_customer'] > 3).astype(int)
df['R_NewMax'] = (df['amount'] > df['customer_max_amount']).astype(int)
df['R_HighCategory'] = (df['zscore_category'] > 2.5).astype(int)

# Pattern-based indicators
df['R_NewMerchantHigh'] = ((df['is_new_merchant'] == 1) & (df['zscore_customer'] > 2)).astype(int)
df['R_HighVelocity'] = (df['velocity_1h_24h'] > 0.5).astype(int)  # High short-term velocity
df['R_ZipMismatch'] = (df['same_zip'] == 0).astype(int)
df['R_HighRiskMerchant'] = (df['merch_fraud_rate'] > 0.1).astype(int)
df['R_UnusualTimeCategory'] = (df['zscore_cat_hour'] > 2).astype(int)

# ========== 11. ONE-HOT ENCODING ==========
print("  11. Encoding categorical variables...")
categorical_cols = ['category']
for col in categorical_cols:
    df[col] = df[col].astype(str)

# One-hot encode categories (keep most frequent ones)
top_categories = df['category'].value_counts().nlargest(20).index
df['category_encoded'] = df['category'].apply(lambda x: x if x in top_categories else 'OTHER')
df_encoded = pd.get_dummies(df[['category_encoded']], prefix='cat', dtype=int)

# ========== 12. FEATURE SELECTION & SCALING ==========
print("  12. Selecting and scaling features...")
# Select the most informative features
num_feats = [
    'amount',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'customer_avg_amount', 'customer_std_amount', 'customer_max_amount',
    'customer_median_amount', 'customer_avg_time_gap',
    'time_since_last', 'amount_per_hour',
    'roll_1h_sum', 'roll_3h_sum', 'roll_24h_sum',
    'roll_1h_mean', 'roll_3h_mean', 'roll_24h_mean',
    'roll_1h_std', 'roll_3h_std', 'roll_24h_std',
    'velocity_1h_24h', 'velocity_3h_24h', 'velocity_1h_3h',
    'acceleration_1h',
    'zscore_customer', 'zscore_category', 'zscore_global',
    'zscore_cust_merch', 'zscore_cat_hour',
    'iqr_score', 'iqr_ratio',
    'merch_fraud_rate', 'merch_risk_score', 'merch_amount_risk',
    'merchant_zip_freq',
    'cust_merch_amount_count', 'cust_merch_amount_mean', 'cust_merch_amount_std',
    'cat_hour_mean', 'cat_hour_std',
    'age_numeric', 'gender_numeric'
]

# Apply RobustScaler (outlier-resistant)
scaler = RobustScaler()
df[num_feats] = scaler.fit_transform(df[num_feats])

# Binary rules
binary_rules = [
    'R_Night', 'R_Weekend', 'R_Rapid', 'R_HighZScore',
    'R_NewMax', 'R_HighCategory', 'R_NewMerchantHigh',
    'R_HighVelocity', 'R_ZipMismatch', 'R_HighRiskMerchant',
    'R_UnusualTimeCategory'
]

# Combine all features
X_omega = pd.concat([df[num_feats], df[binary_rules], df_encoded], axis=1)
y = df['fraud']

print(f"Total Features Generated: {X_omega.shape[1]}")

# =============================================================================
# [STEP B] OPTIMIZED DATA SPLITTING
# =============================================================================
print("\n[STEP B] Creating optimized train/test split...")

X_normal = X_omega[y == 0]
X_fraud = X_omega[y == 1]

# Split normal data for AE training (70%)
X_train_ae, X_temp = train_test_split(X_normal, test_size=0.3, random_state=42)

# Split remaining for EIF training and testing
X_train_eif_normal, X_test_normal = train_test_split(
    X_temp, 
    test_size=0.5, 
    random_state=42
)

# Create test set: normal + all fraud
X_test = pd.concat([X_test_normal, X_fraud], ignore_index=True)
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

# EIF training data (pure normal)
X_train_eif_raw = X_train_eif_normal

print(f"AE Train Size: {len(X_train_ae)}")
print(f"EIF Train Size: {len(X_train_eif_raw)}")
print(f"Test Size: {len(X_test)} (Normal: {len(X_test_normal)}, Fraud: {len(X_fraud)})")

# =============================================================================
# [STEP C] DEEP AUTOENCODER WITH REGULARIZATION
# =============================================================================
print("\n[STEP C] Training Deep Autoencoder...")
input_dim = X_train_ae.shape[1]

# Enhanced architecture
input_layer = Input(shape=(input_dim,))

# Encoder
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)

# Bottleneck
bottleneck = Dense(16, activation='relu', name='bottleneck')(x)

# Decoder
x = Dense(32, activation='relu')(bottleneck)
x = BatchNormalization()(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output_layer = Dense(input_dim, activation='linear')(x)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

# Compile with optimized parameters
autoencoder.compile(
    optimizer=Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    ),
    loss='mse',
    metrics=['mae']
)

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.0001,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1
    )
]

print("Training Autoencoder...")
history = autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.15,
    verbose=1,
    callbacks=callbacks
)

print(f"AE Training completed with final val_loss: {history.history['val_loss'][-1]:.6f}")

# =============================================================================
# [STEP D] FEATURE EXTRACTION FOR EIF+
# =============================================================================
print("\n[STEP D] Extracting features for EIF+...")

def get_enhanced_features(data):
    """Extract enhanced reconstruction features"""
    if hasattr(data, 'values'): 
        data = data.values
    
    # Autoencoder predictions
    rec = autoencoder.predict(data, verbose=0)
    enc = encoder_model.predict(data, verbose=0)
    
    # Multiple reconstruction error metrics
    mse = np.mean(np.power(data - rec, 2), axis=1)
    mae = np.mean(np.abs(data - rec), axis=1)
    rmse = np.sqrt(mse)
    
    # Cosine distance
    norm_a = np.linalg.norm(data, axis=1)
    norm_b = np.linalg.norm(rec, axis=1)
    cos_sim = np.sum(data * rec, axis=1) / (norm_a * norm_b + 1e-10)
    cos_dist = 1 - cos_sim
    
    # Reconstruction ratios
    rec_ratio = np.sum(np.abs(rec), axis=1) / (np.sum(np.abs(data), axis=1) + 1e-10)
    
    # Combine all features
    features = np.hstack([
        enc,  # Bottleneck features (16)
        mse.reshape(-1, 1),
        mae.reshape(-1, 1),
        rmse.reshape(-1, 1),
        cos_dist.reshape(-1, 1),
        rec_ratio.reshape(-1, 1)
    ])
    
    return features, mse

# Extract features for EIF training
X_train_final, mse_train = get_enhanced_features(X_train_eif_raw)

# Calculate adaptive threshold
THRESHOLD = np.percentile(mse_train, AE_PERCENTILE)
print(f"AE Reconstruction Error Threshold (Percentile {AE_PERCENTILE}): {THRESHOLD:.6f}")

# =============================================================================
# [STEP E] TRAIN EXTENDED ISOLATION FOREST (EIF+)
# =============================================================================
print(f"\n[STEP E] Training Extended Isolation Forest (EIF+)...")
print(f"  - n_estimators: {EIF_NTREES}")
print(f"  - max_samples: {EIF_SAMPLE_SIZE}")
print(f"  - max_depth: {EIF_DEPTH}")
print(f"  - plus: True")

# Initialize and train EIF+
eif_model = ExtendedIsolationForest(
    plus=True,
    n_estimators=EIF_NTREES,
    max_samples=EIF_SAMPLE_SIZE,
    max_depth=EIF_DEPTH
)

eif_model.fit(X_train_final)
print("EIF+ model trained successfully!")

# =============================================================================
# [STEP F] OPTIMIZED PREDICTION PIPELINE
# =============================================================================
print("\n[STEP F] Making predictions with optimized pipeline...")

# Extract features for test set
X_test_final, mse_test = get_enhanced_features(X_test)

# Stage 1: Autoencoder filtering
mask_suspect = mse_test > THRESHOLD
final_preds = np.zeros(len(y_test))

suspect_idx = np.where(mask_suspect)[0]
print(f"\nStage 1 - Autoencoder flagged: {len(suspect_idx)}/{len(y_test)} transactions "
      f"({len(suspect_idx)/len(y_test)*100:.2f}%)")

# Stage 2: EIF+ classification
if len(suspect_idx) > 0:
    print("Stage 2 - Extended Isolation Forest (EIF+) evaluation...")
    
    # Get anomaly scores from EIF+
    X_suspect = X_test_final[suspect_idx]
    
    # Calculate anomaly scores
    anomaly_scores = eif_model.predict(X_suspect)
    
    # Normalize scores to 0-1 range
    anomaly_scores_norm = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-10)
    
    # Adaptive thresholding
    score_mean = np.mean(anomaly_scores_norm)
    score_std = np.std(anomaly_scores_norm)
    
    # Dynamic threshold - adjust based on distribution
    eif_threshold = score_mean + 2.5 * score_std
    eif_threshold = min(eif_threshold, 0.8)  # Cap at 0.8
    
    print(f"EIF+ anomaly scores - Mean: {score_mean:.3f}, Std: {score_std:.3f}")
    print(f"EIF+ threshold: {eif_threshold:.3f}")
    
    # Classify as fraud
    confirmed_fraud = anomaly_scores_norm > eif_threshold
    final_preds[suspect_idx[confirmed_fraud]] = 1
    
    print(f"EIF+ confirmed fraud: {np.sum(confirmed_fraud)} transactions")

# =============================================================================
# [STEP G] COMPREHENSIVE EVALUATION
# =============================================================================
print("\n" + "="*60)
print("FINAL OPTIMIZED RESULTS - AE + EIF+")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
TN, FP, FN, TP = cm.ravel()

print("\nConfusion Matrix:")
print("                 Predicted")
print("              Normal   Fraud")
print(f"Actual Normal  {TN:>6}  {FP:>6}")
print(f"Actual Fraud   {FN:>6}  {TP:>6}")

# Detailed classification report
print("\n" + "-"*60)
print("Detailed Classification Report:")
print(classification_report(y_test, final_preds, digits=4, target_names=['Normal', 'Fraud']))

# Key metrics
acc = accuracy_score(y_test, final_preds)
rec = recall_score(y_test, final_preds)
prec = precision_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)
mcc = matthews_corrcoef(y_test, final_preds)

print("\n" + "="*60)
print("KEY PERFORMANCE METRICS:")
print("="*60)
print(f"Accuracy:           {acc:.4f}")
print(f"Precision:          {prec:.4f}")
print(f"Recall:             {rec:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"Matthews Corr Coef: {mcc:.4f}")

# Fraud detection analysis
fraud_detected = TP
total_fraud = FN + TP
fraud_detection_rate = fraud_detected / total_fraud if total_fraud > 0 else 0
false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"\nFraud Detection Rate:  {fraud_detection_rate:.2%}")
print(f"False Positive Rate:   {false_positive_rate:.2%}")
print(f"Total Fraud Detected:  {fraud_detected}/{total_fraud}")

# Performance breakdown
print("\n" + "="*60)
print("PIPELINE PERFORMANCE BREAKDOWN:")
print("="*60)
print(f"AutoEncoder Stage:")
print(f"  - Threshold: {THRESHOLD:.6f} (Percentile {AE_PERCENTILE})")
print(f"  - Flagged: {len(suspect_idx)} transactions")
print(f"  - Filter Efficiency: {len(suspect_idx)/len(y_test)*100:.1f}%")

if len(suspect_idx) > 0:
    print(f"\nEIF+ Stage:")
    print(f"  - Trees: {EIF_NTREES}, Depth: {EIF_DEPTH}")
    print(f"  - Sample Size: {EIF_SAMPLE_SIZE}")
    print(f"  - Confirmed Fraud: {np.sum(final_preds)}")
    print(f"  - Precision on flagged: {TP/len(suspect_idx)*100:.1f}%")

# Performance target assessment
print("\n" + "="*60)
print("PERFORMANCE TARGET ASSESSMENT:")
print("="*60)

if f1 >= 0.90:
    print("🎯 TARGET ACHIEVED: F1-Score >= 0.90!")
    print("   Excellent performance achieved!")
elif f1 >= 0.85:
    print("✅ VERY GOOD: F1-Score >= 0.85")
    print("   Close to target! Try:")
    print("   1. Increase EIF_NTREES to 1000")
    print("   2. Decrease AE_PERCENTILE to 96")
    print("   3. Increase EIF_DEPTH to 50")
elif f1 >= 0.80:
    print("⚠️  GOOD: F1-Score >= 0.80")
    print("   Suggestions for improvement:")
    print("   1. Add more temporal features")
    print("   2. Increase EIF_NTREES to 1200")
    print("   3. Adjust AE architecture (more layers)")
else:
    print("❌ NEEDS IMPROVEMENT: F1-Score < 0.80")
    print("   Try:")
    print("   1. Reduce EIF_SAMPLE_SIZE to 256")
    print("   2. Increase EIF_DEPTH to 60")
    print("   3. Add more fraud-specific features")

# Save models
print("\n[STEP H] Saving models...")
autoencoder.save('optimized_autoencoder_eifplus.h5')
encoder_model.save('optimized_encoder_eifplus.h5')
joblib.dump(scaler, 'robust_scaler_eifplus.pkl')

print("\n✅ Optimization complete! Models saved.")
