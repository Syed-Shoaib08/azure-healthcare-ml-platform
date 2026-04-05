#!/usr/bin/env python
# coding: utf-8

# ## 01_bronze_to_silver
# 
# null

# In[1]:


# ============================================================
# 01_bronze_to_silver.py
# Healthcare ML Platform — Bronze to Silver Transformation
# ============================================================

import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
import io
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# CONFIGURATION 
# ============================================================
STORAGE_ACCOUNT_NAME = "sthealthcareml01"
STORAGE_ACCOUNT_KEY  = "YOUR_ACTUAL_KEY"
BRONZE_CONTAINER     = "bronze-layer"
SILVER_CONTAINER     = "silver-layer"
SOURCE_FILE          = "dataset.csv"
SILVER_FILE          = "silver_icu_patients.parquet"

print("Config loaded.")

# ============================================================
# STEP 1 — Connect to Azure Storage
# ============================================================
connect_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT_NAME};"
    f"AccountKey={STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

blob_service = BlobServiceClient.from_connection_string(connect_str)
print("Connected to Azure Storage.")

# ============================================================
# STEP 2 — Read CSV from Bronze Layer
# ============================================================
bronze_client = blob_service.get_blob_client(
    container=BRONZE_CONTAINER,
    blob=SOURCE_FILE
)

raw_data = bronze_client.download_blob().readall()
df = pd.read_csv(io.BytesIO(raw_data), low_memory=False)

print(f"Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================
# STEP 3 — Understand the Data
# ============================================================
print("\n--- Column Data Types ---")
print(df.dtypes.value_counts())

print(f"\nMissing values: {df.isnull().sum().sum():,} total")
print(f"Duplicate rows: {df.duplicated().sum():,}")

# ============================================================
# STEP 4 — Clean the Data
# ============================================================

# 4a. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace(r'[^\w]', '_', regex=True)
)

# 4b. Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed: {before - len(df):,} rows")

# 4c. Drop columns with >60% missing
thresh = len(df) * 0.4
df = df.dropna(thresh=thresh, axis=1)
print(f"Columns after dropping >60% missing: {df.shape[1]}")

# 4d. Key clinical columns — fill missing with median
clinical_cols = [
    'age', 'bmi', 'heart_rate_apache', 'temp_apache',
    'map_apache', 'resprate_apache', 'spo2_apache',
    'd1_sysbp_max', 'd1_sysbp_min', 'd1_diasbp_max',
    'd1_diasbp_min', 'd1_heartrate_max', 'd1_heartrate_min',
    'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_max',
    'd1_spo2_min', 'd1_temp_max', 'd1_temp_min'
]

for col in clinical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# 4e. Drop rows still missing the target column
if 'hospital_death' in df.columns:
    df = df.dropna(subset=['hospital_death'])
    df['hospital_death'] = df['hospital_death'].astype(int)

print(f"Rows after cleaning: {len(df):,}")

# ============================================================
# STEP 5 — Feature Engineering
# ============================================================

# Shock Index — key sepsis indicator
# Normal range: 0.5-0.7. >1.0 = high sepsis risk
if 'd1_heartrate_max' in df.columns and 'd1_sysbp_min' in df.columns:
    df['shock_index'] = (
        df['d1_heartrate_max'] /
        df['d1_sysbp_min'].replace(0, np.nan)
    ).round(4)
    df['shock_index'] = df['shock_index'].fillna(df['shock_index'].median())

# Age Group
if 'age' in df.columns:
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 40, 60, 75, 120],
        labels=['Young', 'Middle-Age', 'Senior', 'Elderly']
    ).astype(str)

# BMI Category
if 'bmi' in df.columns:
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    ).astype(str)

# Pulse Pressure — indicator of cardiovascular stress
if 'd1_sysbp_max' in df.columns and 'd1_diasbp_min' in df.columns:
    df['pulse_pressure'] = (
        df['d1_sysbp_max'] - df['d1_diasbp_min']
    ).round(2)

# Sepsis Risk Label — our ML target
# Based on: hospital death + shock index + apache score
if 'apache_4a_hospital_death_prob' in df.columns:
    df['sepsis_risk_score'] = (
        (df['apache_4a_hospital_death_prob'] * 0.5) +
        (df.get('shock_index', 0) * 0.3) +
        (df['hospital_death'] * 0.2)
    ).round(4)
    df['sepsis_risk_label'] = (
        df['sepsis_risk_score'] > 0.3
    ).astype(int)
else:
    df['sepsis_risk_label'] = df['hospital_death']

print(f"\nFeature engineering complete.")
print(f"Sepsis positive cases: {df['sepsis_risk_label'].sum():,}")
print(f"Sepsis negative cases: {(df['sepsis_risk_label']==0).sum():,}")
print(f"Total columns now: {df.shape[1]}")

# ============================================================
# STEP 6 — Write Clean Parquet to Silver Layer
# ============================================================

# Convert to parquet in memory
table  = pa.Table.from_pandas(df)
buffer = io.BytesIO()
pq.write_table(table, buffer)
buffer.seek(0)

# Upload to silver layer
silver_client = blob_service.get_blob_client(
    container=SILVER_CONTAINER,
    blob=SILVER_FILE
)
silver_client.upload_blob(buffer, overwrite=True)

print(f"\nSilver layer written: {SILVER_FILE}")
print(f"Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("\n Bronze to Silver complete.")


# In[ ]:




