print("="*60)
print("IMPLEMENTASI AE-EIF+")
print("="*60)

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

dataset2_path = '/Users/apple/Desktop/Skripsi/ae_eif_plus/dataset/bs140513_032310.csv'
output_dir = '/Users/apple/Desktop/Skripsi/ae_eif_plus/data/preprocessed/'
print(f"Loading Dataset 2 from: {dataset2_path}")

df_banksim = pd.read_csv(dataset2_path)

print(f"\n=== DATASET 2 INFORMATION ===")
print(f"Shape: {df_banksim.shape}")
print(f"Columns: {df_banksim.columns.tolist()}")
print(f"\nData types:")
print(df_banksim.dtypes)
print(f"\nFirst 5 rows:")
print(df_banksim.head())

df_banksim = df_banksim.rename(columns={
    'zipcodeOri': 'customer_zip',
    'zipMerchant': 'merchant_zip',
    'fraud': 'fraud'
})

print(f"\n=== TARGET DISTRIBUTION ===")
print(df_banksim['fraud'].value_counts())
print(f"Fraud percentage: {(df_banksim['fraud'].sum() / len(df_banksim)) * 100:.4f}%")

print(f"\n=== UNIQUE VALUES PER COLUMN ===")
for col in df_banksim.columns:
    if df_banksim[col].dtype == 'object':
        unique_count = df_banksim[col].nunique()
        print(f"{col:15s}: {unique_count:6d} unique values")
    else:
        print(f"{col:15s}: Numerical")
        
print(f"\n=== STATISTICAL SUMMARY ===")
print(df_banksim.describe())

print(f"\n=== CLEANING ALL COLUMNS (REMOVING QUOTES) ===")

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

print(f"\n=== SAMPLE DATA AFTER CLEANING ===")
print(df_clean.head(3))

print(f"\n=== DATA TYPES AFTER CLEANING ===")
print(df_clean.dtypes)

print(f"\n=== CONVERTING TO PROPER DATA TYPES ===")

print(f"Converting 'age' to integer...")
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce').fillna(0).astype(int)

print(f"Converting 'amount' to float...")
df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')

print(f"Converting 'fraud' to integer...")
df_clean['fraud'] = pd.to_numeric(df_clean['fraud'], errors='coerce').fillna(0).astype(int)

print(f"\n=== FINAL DATA TYPES ===")
print(df_clean.dtypes)

print(f"\n=== SAMPLE DATA FINAL ===")
print(df_clean.head(5))

print(f"\n=== STATISTICAL SUMMARY ===")
print(df_clean.describe())

print(f"\n=== UNIQUE VALUES PER COLUMN ===")
for col in df_clean.columns:
    unique_count = df_clean[col].nunique()
    sample_values = df_clean[col].unique()[:5] if unique_count > 5 else df_clean[col].unique()
    print(f"{col:15s}: {unique_count:6d} unique | Sample: {sample_values}")

print(f"\n=== TARGET DISTRIBUTION ===")
print(df_clean['fraud'].value_counts())
print(f"Fraud percentage: {(df_clean['fraud'].sum() / len(df_clean)) * 100:.4f}%")

print("\n" + "="*60)
print("FEATURE ENGINEERING FOR CLEANED BANKSIM DATASET")
print("="*60)

def create_banksim_features_fixed(df):
    """
    Create rule-based features for CLEANED BankSim dataset
    """
    df_features = df.copy()
    
    print("Creating comprehensive feature engineering...")
    
    # ==================== 1. RENAME COLUMNS FOR CONSISTENCY ====================
    df_features = df_features.rename(columns={
        'zipcodeOri': 'customer_zip',
        'zipMerchant': 'merchant_zip',
        'fraud': 'fraud'  # Untuk konsistensi
    })
    
    # ==================== 2. TEMPORAL FEATURES ====================
    print("\n1. Temporal Features:")
    
    # a. Transaction hour (assuming 'step' is in hours)
    df_features['transaction_hour'] = df_features['step'] % 24
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['transaction_hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['transaction_hour'] / 24)
    
    # b. Day of week (assuming 24 steps = 1 day)
    df_features['day_of_week'] = (df_features['step'] // 24) % 7
    df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    
    # c. Time of day categories
    df_features['time_of_day'] = pd.cut(df_features['transaction_hour'], 
                                        bins=[0, 6, 12, 18, 24],
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                        include_lowest=True)
    
    # Convert to dummy variables
    time_dummies = pd.get_dummies(df_features['time_of_day'], prefix='time')
    df_features = pd.concat([df_features, time_dummies], axis=1)
    
    print(f"   Created: hour_sin/cos, day_of_week, is_weekend, time_of_day dummies")
    
    # ==================== 3. CUSTOMER BEHAVIOR FEATURES ====================
    print("\n2. Customer Behavior Features:")
    
    # Hitung statistics per customer
    customer_stats = df_features.groupby('customer').agg({
        'amount': ['mean', 'std', 'min', 'max', 'count'],
        'step': ['min', 'max']
    }).reset_index()
    
    customer_stats.columns = ['customer', 'cust_avg_amount', 'cust_std_amount',
                             'cust_min_amount', 'cust_max_amount', 'cust_transaction_count',
                             'cust_first_transaction', 'cust_last_transaction']
    
    # Merge dengan data utama
    df_features = pd.merge(df_features, customer_stats, on='customer', how='left')
    
    # a. Deviation from customer's normal behavior
    df_features['amount_deviation'] = (df_features['amount'] - df_features['cust_avg_amount']) / df_features['cust_std_amount'].replace(0, 1)
    df_features['amount_deviation'] = df_features['amount_deviation'].fillna(0)
    
    # b. Transaction frequency
    df_features['cust_transaction_freq'] = df_features['cust_transaction_count'] / (df_features['cust_last_transaction'] - df_features['cust_first_transaction'] + 1)
    
    # c. Time since last transaction (per customer)
    df_features['time_since_last'] = df_features.groupby('customer')['step'].diff().fillna(24)
    
    print(f"   Created: customer stats, amount deviation, transaction frequency, time since last")
    
    # ==================== 4. DEMOGRAPHIC FEATURES ====================
    print("\n3. Demographic Features:")
    
    # a. Age groups (FIXED - sekarang 'age' sudah integer)
    df_features['age_group'] = pd.cut(df_features['age'], 
                                      bins=[0, 25, 35, 45, 55, 65, 100],
                                      labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
    
    age_dummies = pd.get_dummies(df_features['age_group'], prefix='age')
    df_features = pd.concat([df_features, age_dummies], axis=1)
    
    # b. Gender encoding (FIXED - sudah cleaned)
    df_features['gender_M'] = (df_features['gender'] == 'M').astype(int)
    df_features['gender_F'] = (df_features['gender'] == 'F').astype(int)
    df_features['gender_U'] = (df_features['gender'] == 'U').astype(int)
    
    # c. Age-risk interaction
    amount_q75 = df_features['amount'].quantile(0.75)
    df_features['young_high_amount'] = ((df_features['age'] < 30) & (df_features['amount'] > amount_q75)).astype(int)
    
    print(f"   Created: age groups, gender encoding, age-risk interactions")
    
    # ==================== 5. MERCHANT & CATEGORY FEATURES ====================
    print("\n4. Merchant & Category Features:")
    
    # a. Merchant risk score
    merchant_fraud_rate = df_features.groupby('merchant')['fraud'].mean()
    df_features['merchant_fraud_rate'] = df_features['merchant'].map(merchant_fraud_rate).fillna(0)
    
    # b. Category encoding
    category_dummies = pd.get_dummies(df_features['category'], prefix='category')
    df_features = pd.concat([df_features, category_dummies], axis=1)
    
    # c. Unusual category for customer
    customer_categories = df_features.groupby('customer')['category'].apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').to_dict()
    df_features['usual_category'] = df_features.apply(
        lambda row: 1 if row['category'] == customer_categories.get(row['customer'], 'unknown') else 0,
        axis=1
    )
    
    print(f"   Created: merchant fraud rate, category dummies, usual category")
    
    # ==================== 6. AMOUNT-BASED FEATURES ====================
    print("\n5. Amount-Based Features:")
    
    # a. Amount categories
    df_features['amount_category'] = pd.qcut(df_features['amount'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    amount_dummies = pd.get_dummies(df_features['amount_category'], prefix='amount_cat')
    df_features = pd.concat([df_features, amount_dummies], axis=1)
    
    # b. Amount outliers
    amount_mean = df_features['amount'].mean()
    amount_std = df_features['amount'].std()
    df_features['amount_zscore'] = (df_features['amount'] - amount_mean) / amount_std
    df_features['high_amount'] = (df_features['amount_zscore'] > 3).astype(int)
    df_features['very_low_amount'] = (df_features['amount_zscore'] < -3).astype(int)
    
    # c. Amount relative to customer history
    df_features['amount_vs_cust_avg'] = df_features['amount'] / df_features['cust_avg_amount'].replace(0, 1)
    
    print(f"   Created: amount categories, z-score, high/low flags, relative amount")
    
    # ==================== 7. GEOGRAPHICAL FEATURES ====================
    print("\n6. Geographical Features:")
    
    # a. Zipcode features (FIXED - sudah cleaned)
    df_features['customer_zip_prefix'] = df_features['customer_zip'].astype(str).str[:3]
    df_features['merchant_zip_prefix'] = df_features['merchant_zip'].astype(str).str[:3]
    df_features['same_zip_area'] = (df_features['customer_zip_prefix'] == df_features['merchant_zip_prefix']).astype(int)
    
    # b. Unusual location
    customer_zip_history = df_features.groupby('customer')['customer_zip_prefix'].apply(set).to_dict()
    df_features['new_zip_area'] = df_features.apply(
        lambda row: 0 if row['merchant_zip_prefix'] in customer_zip_history.get(row['customer'], set()) else 1,
        axis=1
    )
    
    print(f"   Created: same zip area, new zip area")
    
    # ==================== 8. RULE-BASED COMBINATION FEATURES ====================
    print("\n7. Rule-Based Combination Features:")
    
    # Combination rules
    df_features['high_amount_new_location'] = (df_features['high_amount'] & df_features['new_zip_area']).astype(int)
    df_features['night_transaction_high_amount'] = ((df_features['time_of_day'] == 'Night') & df_features['high_amount']).astype(int)
    df_features['weekend_unusual_category'] = (df_features['is_weekend'] & (df_features['usual_category'] == 0)).astype(int)
    df_features['young_new_merchant'] = ((df_features['age'] < 25) & df_features['new_zip_area']).astype(int)
    
    print(f"   Created: combination rules")
    
    # ==================== 9. FINAL CLEANUP ====================
    print("\n8. Final Cleanup:")
    
    # Drop helper columns
    cols_to_drop = ['time_of_day', 'age_group', 'amount_category', 
                   'customer_zip_prefix', 'merchant_zip_prefix',
                   'transaction_hour', 'day_of_week', 'gender']
    
    df_features = df_features.drop(columns=cols_to_drop, errors='ignore')
    
    # Identify rule-based features
    rule_based_patterns = ['high_amount', 'very_low_amount', 'same_zip_area', 'new_zip_area',
                          'is_weekend', 'young_high_amount', 'usual_category',
                          'high_amount_new_location', 'night_transaction_high_amount',
                          'weekend_unusual_category', 'young_new_merchant',
                          'gender_', 'time_', 'age_', 'category_', 'amount_cat_']
    
    rule_based_features = [col for col in df_features.columns if any(pattern in col for pattern in rule_based_patterns)]
    
    print(f"   Total features: {df_features.shape[1]}")
    print(f"   Rule-based features: {len(rule_based_features)}")
    
    return df_features, rule_based_features

# Apply feature engineering
df_banksim_enhanced, rule_based_features = create_banksim_features_fixed(df_clean)

print(f"\n" + "="*60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("="*60)

print(f"\n=== FINAL DATASET SHAPE ===")
print(f"Original shape: {df_banksim.shape}")
print(f"Enhanced shape: {df_banksim_enhanced.shape}")
print(f"Features added: {df_banksim_enhanced.shape[1] - df_banksim.shape[1]}")

print(f"\n=== SAMPLE OF ENHANCED DATA ===")
print("First 5 rows, first 10 columns:")
print(df_banksim_enhanced.iloc[:5, :10])

print(f"\n=== RULE-BASED FEATURES CREATED ===")
print(f"Total: {len(rule_based_features)}")
for i, feat in enumerate(rule_based_features[:20]):  # Show first 20
    print(f"  {i+1:2d}. {feat}")
if len(rule_based_features) > 20:
    print(f"  ... and {len(rule_based_features) - 20} more")

# Save enhanced dataset
enhanced_path = os.path.join(output_dir, 'banksim_enhanced.pkl')
import joblib

banksim_enhanced_data = {
    'df': df_banksim_enhanced,
    'rule_based_features': rule_based_features,
    'dataset_info': {
        'original_samples': len(df_banksim),
        'enhanced_samples': len(df_banksim_enhanced),
        'total_features': df_banksim_enhanced.shape[1],
        'original_features': df_banksim.shape[1],
        'engineered_features': df_banksim_enhanced.shape[1] - df_banksim.shape[1],
        'fraud_rate': df_banksim_enhanced['fraud'].mean() * 100
    }
}

joblib.dump(banksim_enhanced_data, enhanced_path)
print(f"\nEnhanced dataset saved to: {enhanced_path}")

banksim_data = joblib.load(os.path.join(output_dir, 'banksim_enhanced.pkl'))
df_enhanced = banksim_data['df']

print("=== DATASET LOADED ===")
print(f"Dataset shape: {df_enhanced.shape}")
print(f"Features: {df_enhanced.shape[1] - 1}") 
print(f"Fraud rate: {df_enhanced['fraud'].mean()*100:.4f}%")
print(f"Fraud samples: {df_enhanced['fraud'].sum():,}")

# Separate features and target
X = df_enhanced.drop('fraud', axis=1)
y = df_enhanced['fraud']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ==================== DATA PREPROCESSING ====================
print("\n" + "="*60)
print("DATA PREPROCESSING FOR AUTOENCODER")
print("="*60)

print("\n1. Handling categorical features...")
# Identify categorical columns (non-numeric)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"   Categorical columns: {categorical_cols}")

if categorical_cols:
    # Label encoding for categorical features
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded '{col}' - {len(le.classes_)} classes")
else:
    print("   No categorical columns found")

print(f"\n2. Feature scaling...")
# Use RobustScaler to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Original data range:")
print(f"     Min: {X.values.min():.4f}, Max: {X.values.max():.4f}")
print(f"     Mean: {X.values.mean():.4f}, Std: {X.values.std():.4f}")

print(f"\n   Scaled data range:")
print(f"     Min: {X_scaled.min():.4f}, Max: {X_scaled.max():.4f}")
print(f"     Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")


# ==================== DATA SPLITTING (SESUAI JURNAL) ====================
print("\n" + "="*60)
print("DATA SPLITTING ACCORDING TO JOURNAL")
print("="*60)

print("\nAccording to journal (Section 7.2.2):")
print("- AutoEncoder trained ONLY on NORMAL transactions")
print("- Training: 70% of normal transactions")
print("- Threshold determination: Remaining 30% of training set")
print("- EIF training: Uses threshold-determined data")

# Separate normal and fraud data
X_normal = X_scaled[y == 0]
X_fraud = X_scaled[y == 1]
y_normal = y[y == 0]
y_fraud = y[y == 1]

print(f"\nNormal transactions: {len(X_normal):,}")
print(f"Fraud transactions: {len(X_fraud):,}")
print(f"Fraud percentage: {len(X_fraud)/len(X_scaled)*100:.4f}%")

# Split normal data: 70% for AE training, 30% for threshold determination
X_train_normal, X_temp_normal = train_test_split(
    X_normal, test_size=0.3, random_state=42
)

print(f"\nAE Training data (normal only): {len(X_train_normal):,}")
print(f"Threshold determination data: {len(X_temp_normal):,}")

# Test set: 30% normal + all fraud
X_test_normal = X_temp_normal
X_test = np.vstack([X_test_normal, X_fraud])
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_fraud))])

print(f"\nTest data:")
print(f"  Normal in test: {len(X_test_normal):,}")
print(f"  Fraud in test: {len(X_fraud):,}")
print(f"  Total test: {len(X_test):,}")
print(f"  Test fraud rate: {y_test.mean()*100:.4f}%")

print("\n" + "="*60)
print("AUTOENCODER ARCHITECTURE (According to Section 6.1)")
print("="*60)

print("\nJournal specifications:")
print("- Input layer: n features")
print("- Hidden layers: 14 → 7 → 14 nodes")  
print("- Output layer: n features (reconstruction)")
print("- Activation: ReLU")
print("- Training: Only on normal transactions")

input_dim = X_train_normal.shape[1]
encoding_dim1 = 14  # First hidden layer
encoding_dim2 = 7   # Bottleneck layer (compressed representation)
encoding_dim3 = 14  # Third hidden layer

print(f"\nAutoEncoder Configuration for BankSim:")
print(f"Input dimension: {input_dim} features")
print(f"Architecture: Input({input_dim}) → Dense({encoding_dim1}) → Dense({encoding_dim2})")
print(f"               → Dense({encoding_dim3}) → Output({input_dim})")
print(f"Training samples: {len(X_train_normal):,} (normal only)")

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Build AutoEncoder
print("\nBuilding AutoEncoder model...")

input_layer = Input(shape=(input_dim,), name='input')

# Encoder
encoded1 = Dense(encoding_dim1, activation='relu', name='encoder1')(input_layer)
encoded1_bn = BatchNormalization(name='bn1')(encoded1)
encoded1_drop = Dropout(0.1, name='dropout1')(encoded1_bn)

encoded2 = Dense(encoding_dim2, activation='relu', name='encoder2')(encoded1_drop)
encoded2_bn = BatchNormalization(name='bn2')(encoded2)  # Bottleneck

encoded3 = Dense(encoding_dim3, activation='relu', name='encoder3')(encoded2_bn)
encoded3_bn = BatchNormalization(name='bn3')(encoded3)
encoded3_drop = Dropout(0.1, name='dropout2')(encoded3_bn)

# Decoder
decoded1 = Dense(encoding_dim2, activation='relu', name='decoder1')(encoded3_drop)
decoded1_bn = BatchNormalization(name='bn4')(decoded1)

decoded2 = Dense(encoding_dim1, activation='relu', name='decoder2')(decoded1_bn)
decoded2_bn = BatchNormalization(name='bn5')(decoded2)

output_layer = Dense(input_dim, activation='linear', name='output')(decoded2_bn)

# Create models
autoencoder = Model(inputs=input_layer, outputs=output_layer, name='AutoEncoder')
encoder = Model(inputs=input_layer, outputs=encoded2_bn, name='Encoder')  # For μ feature extraction

print("\nAutoEncoder Summary:")
autoencoder.summary()

print(f"\nEncoder (for μ features) Summary:")
encoder.summary()

# ==================== TRAINING AUTOENCODER ====================
print("\n" + "="*60)
print("TRAINING AUTOENCODER (Section 6.3.1)")
print("="*60)

# Compile model
autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error for reconstruction
    metrics=['mae', tf.keras.metrics.CosineSimilarity()]
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(output_dir, 'banksim_autoencoder_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print(f"\nTraining configuration:")
print(f"- Epochs: 100")
print(f"- Batch size: 32")
print(f"- Validation split: 0.2")
print(f"- Early stopping patience: 15")
print(f"- Learning rate reduction patience: 7")

# Train AutoEncoder
print("\nStarting AutoEncoder training...")
history = autoencoder.fit(
    X_train_normal, X_train_normal,  # Autoencoder learns to reconstruct its input
    epochs=15,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\nAutoEncoder training completed!")

# Save final model
autoencoder.save(os.path.join(output_dir, 'banksim_autoencoder_final.h5'))
encoder.save(os.path.join(output_dir, 'banksim_encoder_final.h5'))
print(f"\nModels saved to {output_dir}")


print("\n" + "="*60)
print("TRAINING RESULTS VISUALIZATION")
print("="*60)

# Plot training history
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Loss (MSE)
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 2. MAE
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Cosine Similarity
axes[2].plot(history.history['cosine_similarity'], label='Training Similarity', linewidth=2)
axes[2].plot(history.history['val_cosine_similarity'], label='Validation Similarity', linewidth=2)
axes[2].set_title('Cosine Similarity', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Similarity')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final metrics
print("\n=== FINAL TRAINING METRICS ===")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"Final training MAE: {history.history['mae'][-1]:.6f}")
print(f"Final validation MAE: {history.history['val_mae'][-1]:.6f}")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
print(f"Best epoch: {history.history['val_loss'].index(min(history.history['val_loss'])) + 1}")


print("\n" + "="*60)
print("EXTRACTING μ FEATURES (Section 6.3.2)")
print("="*60)

print("\nAccording to journal:")
print("μ = [A, B, C] where:")
print("A: Compressed representation from AutoEncoder")
print("B: Euclidean distance between input and reconstruction")
print("C: Similarity degree between input and reconstruction")

# Get reconstructions for test data
print(f"\n1. Generating reconstructions for test data...")
reconstructions = autoencoder.predict(X_test, verbose=1)

# Calculate reconstruction errors
print(f"\n2. Calculating reconstruction errors...")
mse_per_sample = np.mean(np.square(X_test - reconstructions), axis=1)
mae_per_sample = np.mean(np.abs(X_test - reconstructions), axis=1)

print(f"   Reconstruction error statistics:")
print(f"   MSE - Min: {mse_per_sample.min():.6f}, Max: {mse_per_sample.max():.6f}")
print(f"        Mean: {mse_per_sample.mean():.6f}, Std: {mse_per_sample.std():.6f}")

# Calculate μ features
print(f"\n3. Calculating μ features...")

# A: Compressed representation (from bottleneck layer)
print("   A: Getting compressed representation...")
A_features = encoder.predict(X_test, verbose=1)  # Output from bottleneck (7 dimensions)

# B: Euclidean distance (reconstruction error)
print("   B: Calculating Euclidean distance...")
B_features = np.sqrt(np.sum(np.square(X_test - reconstructions), axis=1, keepdims=True))

# C: Similarity degree (cosine similarity)
print("   C: Calculating cosine similarity...")
from sklearn.metrics.pairwise import cosine_similarity

C_features = np.array([cosine_similarity([X_test[i]], [reconstructions[i]])[0][0] 
                       for i in range(len(X_test))])
C_features = C_features.reshape(-1, 1)

print(f"\nμ features extracted:")
print(f"A (compressed): shape {A_features.shape}")
print(f"B (Euclidean):  shape {B_features.shape}")
print(f"C (similarity): shape {C_features.shape}")
print(f"Total μ features per sample: {A_features.shape[1] + B_features.shape[1] + C_features.shape[1]}")

# Combine all μ features
mu_features = np.hstack([A_features, B_features, C_features])
print(f"\nCombined μ features shape: {mu_features.shape}")

# Save μ features
mu_features_path = os.path.join(output_dir, 'banksim_mu_features.pkl')
joblib.dump({
    'A_features': A_features,
    'B_features': B_features,
    'C_features': C_features,
    'mu_features': mu_features,
    'reconstruction_errors': mse_per_sample,
    'reconstructions': reconstructions
}, mu_features_path)
print(f"\nμ features saved to: {mu_features_path}")


print("\n" + "="*60)
print("THRESHOLD DETERMINATION (Section 6.3.1)")
print("="*60)

print("\nAccording to journal:")
print("- Calculate reconstruction errors on training data (normal only)")
print("- Determine threshold based on percentile")
print("- Flag transactions above threshold as suspicious")

# Calculate reconstruction errors on training data
print("\n1. Calculating reconstruction errors on training data (normal only)...")
train_reconstructions = autoencoder.predict(X_train_normal, verbose=0)
train_mse = np.mean(np.square(X_train_normal - train_reconstructions), axis=1)

# Determine threshold (sesuai jurnal: 95th percentile)
threshold_percentile = 95
threshold = np.percentile(train_mse, threshold_percentile)

print(f"\n2. Threshold determination:")
print(f"   Using {threshold_percentile}th percentile of training reconstruction errors")
print(f"   Threshold value: {threshold:.6f}")
print(f"   Training MSE statistics:")
print(f"     Mean: {train_mse.mean():.6f}")
print(f"     Std:  {train_mse.std():.6f}")
print(f"     Min:  {train_mse.min():.6f}")
print(f"     Max:  {train_mse.max():.6f}")
print(f"     {threshold_percentile}th percentile: {threshold:.6f}")

# Identify suspicious transactions
suspicious_mask = mse_per_sample > threshold
normal_mask = mse_per_sample <= threshold

print(f"\n3. Suspicious transaction identification:")
print(f"   Total test transactions: {len(X_test):,}")
print(f"   Suspicious transactions (AE): {suspicious_mask.sum():,} ({suspicious_mask.sum()/len(X_test)*100:.2f}%)")
print(f"   Normal transactions (AE): {normal_mask.sum():,} ({normal_mask.sum()/len(X_test)*100:.2f}%)")

print(f"\n   Actual fraud detected as suspicious: {(suspicious_mask & (y_test == 1)).sum():,}")
print(f"   Actual fraud missed: {(normal_mask & (y_test == 1)).sum():,}")
print(f"   False positives (normal flagged as suspicious): {(suspicious_mask & (y_test == 0)).sum():,}")

# Visualize threshold
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of training reconstruction errors
axes[0].hist(train_mse, bins=50, alpha=0.7, edgecolor='black', density=True)
axes[0].axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold_percentile}th percentile)')
axes[0].set_title('Training Reconstruction Errors\n(Normal Transactions Only)', fontweight='bold')
axes[0].set_xlabel('Reconstruction Error (MSE)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test data errors by class
test_mse_normal = mse_per_sample[y_test == 0]
test_mse_fraud = mse_per_sample[y_test == 1]

box_data = [test_mse_normal, test_mse_fraud]
axes[1].boxplot(box_data, labels=['Normal', 'Fraud'])
axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold_percentile}th percentile)')
axes[1].set_title('Test Reconstruction Errors by Class', fontweight='bold')
axes[1].set_ylabel('Reconstruction Error (MSE)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("PREPARING DATA FOR EXTENDED ISOLATION FOREST")
print("="*60)

print("\nAccording to journal Section 6.3.2:")
print("- EIF uses Ω (original) + μ (AE) features")
print("- EIF analyzes only suspicious transactions from AE")
print("- Ω features: original + rule-based features")
print("- μ features: A, B, C from AutoEncoder")

# Ω features: Original + Rule-based features (scaled)
omega_features = X_test  # Already scaled

print(f"\nΩ features (original + rule-based): shape {omega_features.shape}")

# Combine Ω + μ features
X_eif_input = np.hstack([omega_features, mu_features])

print(f"\nCombined features for EIF (Ω + μ):")
print(f"Ω features: {omega_features.shape[1]} features")
print(f"μ features: {mu_features.shape[1]} features")
print(f"Total features: {X_eif_input.shape[1]} features")

# Split data based on AE threshold
X_suspicious = X_eif_input[suspicious_mask]  # Untuk EIF analysis
X_normal_ae = X_eif_input[normal_mask]       # Sudah diklasifikasi normal oleh AE
y_suspicious = y_test[suspicious_mask]
y_normal_ae = y_test[normal_mask]

print(f"\nData split for AE-EIF pipeline:")
print(f"1. AE Normal (not sent to EIF): {X_normal_ae.shape[0]:,} samples")
print(f"2. AE Suspicious (sent to EIF): {X_suspicious.shape[0]:,} samples")
print(f"   - Actual fraud in suspicious: {y_suspicious.sum():,}")
print(f"   - Actual normal in suspicious: {(y_suspicious == 0).sum():,}")

# Save AE-EIF pipeline data
ae_eif_data = {
    # AutoEncoder models
    'autoencoder': autoencoder,
    'encoder': encoder,
    
    # Threshold information
    'threshold': threshold,
    'threshold_percentile': threshold_percentile,
    'suspicious_mask': suspicious_mask,
    'normal_mask': normal_mask,
    
    # μ features
    'A_features': A_features,
    'B_features': B_features,
    'C_features': C_features,
    'mu_features': mu_features,
    
    # Combined features
    'X_eif_input': X_eif_input,
    'X_suspicious': X_suspicious,
    'X_normal_ae': X_normal_ae,
    'y_suspicious': y_suspicious,
    'y_normal_ae': y_normal_ae,
    
    # Original data
    'X_test': X_test,
    'y_test': y_test,
    'reconstruction_errors': mse_per_sample,
    
    # Training history
    'history': history.history,
    
    # Metadata
    'metadata': {
        'dataset': 'BankSim',
        'input_dim': input_dim,
        'encoding_dims': [encoding_dim1, encoding_dim2, encoding_dim3],
        'total_test_samples': len(X_test),
        'suspicious_samples': suspicious_mask.sum(),
        'normal_samples_ae': normal_mask.sum(),
        'fraud_in_suspicious': y_suspicious.sum(),
        'fraud_in_normal_ae': (normal_mask & (y_test == 1)).sum(),
        'fraud_detection_rate_ae': (suspicious_mask & (y_test == 1)).sum() / y_test.sum() * 100,
        'false_positive_rate_ae': (suspicious_mask & (y_test == 0)).sum() / (y_test == 0).sum() * 100
    }
}

# Save to file
ae_eif_pipeline_path = os.path.join(output_dir, 'banksim_ae_eif_pipeline.pkl')
joblib.dump(ae_eif_data, ae_eif_pipeline_path)

print(f"\n=== AE-EIF PIPELINE DATA SAVED ===")
print(f"File: {ae_eif_pipeline_path}")

print(f"\n=== AUTOENCODER PERFORMANCE SUMMARY ===")
meta = ae_eif_data['metadata']
print(f"✅ 1. AutoEncoder trained successfully on {len(X_train_normal):,} normal transactions")
print(f"   - Architecture: {input_dim} → {encoding_dim1} → {encoding_dim2} → {encoding_dim3} → {input_dim}")
print(f"   - Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"\n✅ 2. μ features extracted:")
print(f"   - A (compressed): {A_features.shape[1]} features")
print(f"   - B (Euclidean): {B_features.shape[1]} feature")
print(f"   - C (similarity): {C_features.shape[1]} feature")
print(f"   - Total μ: {mu_features.shape[1]} features")
print(f"\n✅ 3. Threshold determined:")
print(f"   - Threshold: {threshold:.6f} ({threshold_percentile}th percentile)")
print(f"   - Suspicious transactions: {meta['suspicious_samples']:,} ({meta['suspicious_samples']/meta['total_test_samples']*100:.1f}%)")
print(f"\n✅ 4. Fraud detection by AE:")
print(f"   - Fraud detected: {meta['fraud_in_suspicious']:,} out of {y_test.sum():,}")
print(f"   - Detection rate: {meta['fraud_detection_rate_ae']:.1f}%")
print(f"   - False positive rate: {meta['false_positive_rate_ae']:.1f}%")

print(f"\n" + "="*60)
print("NEXT STEP: EXTENDED ISOLATION FOREST (EIF)")
print("="*60)
print("\nReady to implement Extended Isolation Forest with:")
print(f"1. Ω features: {omega_features.shape[1]} (original + rule-based)")
print(f"2. μ features: {mu_features.shape[1]} (from AutoEncoder)")
print(f"3. Total features for EIF: {X_eif_input.shape[1]}")
print(f"4. Suspicious transactions to analyze: {X_suspicious.shape[0]:,}")


print("\n=== TRAINING HISTORY VISUALIZATION ===")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Loss (MSE)
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. MAE
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Cosine Similarity
axes[1, 0].plot(history.history['cosine_similarity'], label='Training Similarity', linewidth=2)
axes[1, 0].plot(history.history['val_cosine_similarity'], label='Validation Similarity', linewidth=2)
axes[1, 0].set_title('Cosine Similarity', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Similarity')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Learning Rate
if 'lr' in history.history:
    axes[1, 1].plot(history.history['lr'], linewidth=2, color='purple')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.show()

# Save final model
autoencoder.save(os.path.join(output_dir, 'autoencoder_final.h5'))
encoder.save(os.path.join(output_dir, 'encoder_final.h5'))
print(f"\nModels saved to {output_dir}")


print("\n=== TRAINING HISTORY VISUALIZATION ===")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Loss (MSE)
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. MAE
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Cosine Similarity
axes[1, 0].plot(history.history['cosine_similarity'], label='Training Similarity', linewidth=2)
axes[1, 0].plot(history.history['val_cosine_similarity'], label='Validation Similarity', linewidth=2)
axes[1, 0].set_title('Cosine Similarity', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Similarity')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Learning Rate
if 'lr' in history.history:
    axes[1, 1].plot(history.history['lr'], linewidth=2, color='purple')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.show()

# Save final model
autoencoder.save(os.path.join(output_dir, 'autoencoder_final.h5'))
encoder.save(os.path.join(output_dir, 'encoder_final.h5'))
print(f"\nModels saved to {output_dir}")


print("="*60)
print("IMPLEMENTING AE-EIF HYBRID MODEL")
print("="*60)

print("\n=== LOADING PREPROCESSED DATA ===")

import numpy as np
import pandas as pd
import joblib
import os
from model_reboot import *

# Load preprocessed data
output_dir = '/Users/apple/Desktop/Skripsi/ExIFFI/data/preprocessed/'

# Load AE-EIF data
ae_eif_data = joblib.load(os.path.join(output_dir, 'banksim_ae_eif_pipeline.pkl'))

# Load original preprocessed data
preprocessed_data = joblib.load(os.path.join(output_dir, 'banksim_mu_features.pkl'))

# Extract data
X_suspicious = ae_eif_data['X_suspicious']  # Ω + μ features untuk suspicious transactions
y_suspicious = ae_eif_data['y_suspicious']  # Labels untuk suspicious transactions
X_normal_ae = ae_eif_data['X_normal_ae']    # Transaksi yang sudah diklasifikasi normal oleh AE
y_normal_ae = ae_eif_data['y_normal_ae']    # Labels untuk normal transactions

threshold = ae_eif_data['threshold']
suspicious_mask = ae_eif_data['suspicious_mask']

print(f"Data loaded:")
print(f"X_suspicious shape (Ω+μ): {X_suspicious.shape}")
print(f"y_suspicious shape: {y_suspicious.shape}")
print(f"X_normal_ae shape: {X_normal_ae.shape}")
print(f"y_normal_ae shape: {y_normal_ae.shape}")
print(f"AE Threshold: {threshold:.6f}")
print(f"\nSuspicious transactions (to EIF): {len(X_suspicious):,}")
print(f"Normal transactions (from AE): {len(X_normal_ae):,}")
print(f"Fraud in suspicious: {y_suspicious.sum():,}")
print(f"Normal in suspicious: {(y_suspicious == 0).sum():,}")


