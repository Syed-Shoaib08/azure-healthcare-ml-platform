#!/usr/bin/env python
# coding: utf-8

# ## 02_silver_to_gold
# 
# null

# In[1]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install imbalanced-learn


# In[2]:


# ============================================================
# 02_silver_to_gold.py
# Healthcare ML Platform — Silver to Gold + SMOTE Expansion
# ============================================================

import pandas as pd
import numpy as np
import io
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIGURATION
# ============================================================
STORAGE_ACCOUNT_NAME = "sthealthcareml01"
STORAGE_ACCOUNT_KEY  = "YOUR_ACTUAL_KEY"
SILVER_CONTAINER     = "silver-layer"
GOLD_CONTAINER       = "gold-layer"
SILVER_FILE          = "silver_icu_patients.parquet"
GOLD_FILE            = "gold_ml_ready.parquet"
TARGET_ROWS          = 20_000_000
TARGET_COL           = "sepsis_risk_label"

print("Config loaded.")

# ============================================================
# STEP 1 — Connect + Read Silver Data
# ============================================================
connect_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT_NAME};"
    f"AccountKey={STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

blob_service = BlobServiceClient.from_connection_string(connect_str)

silver_client = blob_service.get_blob_client(
    container=SILVER_CONTAINER,
    blob=SILVER_FILE
)

data   = silver_client.download_blob().readall()
df     = pq.read_table(io.BytesIO(data)).to_pandas()

print(f"Silver data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")

# ============================================================
# STEP 2 — Feature Selection: Remove Non-Numeric + Target
# ============================================================
# Separate target
y  = df[TARGET_COL].astype(int)
X  = df.drop(columns=[TARGET_COL])

# Keep only numeric columns
X  = X.select_dtypes(include=[np.number])
print(f"\nNumeric columns for selection: {X.shape[1]}")

# Fill any remaining nulls
X  = X.fillna(X.median())

# ============================================================
# STEP 3 — Variance Threshold Filter
# Remove columns where value barely changes
# ============================================================
vt      = VarianceThreshold(threshold=0.01)
vt.fit(X)
X       = X[X.columns[vt.get_support()]]
print(f"After variance filter: {X.shape[1]} columns")

# ============================================================
# STEP 4 — Correlation Filter
# Remove columns that are >92% correlated with each other
# ============================================================
corr_matrix  = X.corr().abs()
upper        = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop      = [
    col for col in upper.columns
    if any(upper[col] > 0.92)
]
X            = X.drop(columns=to_drop)
print(f"After correlation filter: {X.shape[1]} columns")
print(f"Dropped correlated columns: {len(to_drop)}")

# ============================================================
# STEP 5 — Random Forest Feature Importance
# Train on sample for speed, rank all features
# ============================================================
print("\nTraining Random Forest for feature importance...")
print("(This may take 2-3 minutes)")

# Use sample for speed on 91K dataset
sample_size = min(20000, len(X))
X_sample    = X.sample(n=sample_size, random_state=42)
y_sample    = y[X_sample.index]

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_sample, y_sample)

# Get importance scores
importance_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features by importance:")
print(importance_df.head(15).to_string(index=False))

# ============================================================
# STEP 6 — Clinical Domain Override
# Force-keep these regardless of importance score
# ============================================================
clinical_must_keep = [
    'apache_4a_hospital_death_prob',
    'shock_index',
    'age',
    'bmi',
    'heart_rate_apache',
    'map_apache',
    'resprate_apache',
    'spo2_apache',
    'temp_apache',
    'pulse_pressure'
]

# Keep top 20 by importance + clinical overrides
top_features     = importance_df.head(20)['feature'].tolist()
clinical_present = [c for c in clinical_must_keep if c in X.columns]
final_features   = list(set(top_features + clinical_present))

X_final = X[final_features]
print(f"\nFinal features selected: {len(final_features)}")
print(f"Features: {sorted(final_features)}")

# ============================================================
# STEP 7 — SMOTE Expansion to 20M rows
# ============================================================
print(f"\nStarting SMOTE expansion...")
print(f"Current: {len(X_final):,} rows")
print(f"Target:  {TARGET_ROWS:,} rows")
print("(This will take several minutes — large expansion)")

# Calculate sampling strategy
# We want total rows = TARGET_ROWS with balanced classes
target_per_class = TARGET_ROWS // 2

smote = SMOTE(
    sampling_strategy={
        0: target_per_class,
        1: target_per_class
    },
    random_state=42,
    k_neighbors=5,
)

X_resampled, y_resampled = smote.fit_resample(X_final, y)

print(f"\nSMOTE complete!")
print(f"Expanded rows: {len(X_resampled):,}")
print(f"Class distribution:\n{pd.Series(y_resampled).value_counts()}")

# ============================================================
# STEP 8 — Build Final Gold DataFrame
# ============================================================
df_gold                    = pd.DataFrame(
    X_resampled,
    columns=final_features
)
df_gold[TARGET_COL]        = y_resampled
df_gold['record_id']       = range(1, len(df_gold) + 1)

print(f"\nGold dataset: {df_gold.shape[0]:,} rows, {df_gold.shape[1]} columns")

# ============================================================
# STEP 9 — Write to Gold Layer in chunks
# SMOTE data is large — write in 5 partitions
# ============================================================
print("\nWriting Gold layer in partitions...")

chunk_size  = len(df_gold) // 5
chunks      = [
    df_gold.iloc[i:i+chunk_size]
    for i in range(0, len(df_gold), chunk_size)
]

for idx, chunk in enumerate(chunks):
    table      = pa.Table.from_pandas(chunk, preserve_index=False)
    buffer     = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)

    filename   = f"gold_ml_ready_part{idx+1}.parquet"
    gold_client = blob_service.get_blob_client(
        container=GOLD_CONTAINER,
        blob=filename
    )
    gold_client.upload_blob(buffer, overwrite=True)
    print(f"  Part {idx+1}/5 written — {len(chunk):,} rows")

print(f"\n Silver to Gold complete.")
print(f"Total rows in Gold: {df_gold.shape[0]:,}")
print(f"Total columns: {df_gold.shape[1]}")
print(f"Files written: 5 partitions in gold-layer container")


# In[ ]:




