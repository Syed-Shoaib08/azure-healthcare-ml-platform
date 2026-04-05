#!/usr/bin/env python
# coding: utf-8

# ## 04_score_and_predict
# 
# null

# In[10]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install azure-ai-ml azure-identity xgboost pyarrow azure-storage-blob


# In[11]:


# ============================================================
# 04_score_and_predict.py
# Healthcare ML Platform — Sepsis Risk Scoring
# ============================================================

import pandas as pd
import numpy as np
import io
import os
import xgboost as xgb
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

# ============================================================
# CONFIGURATION
# ============================================================
STORAGE_ACCOUNT_NAME = "sthealthcareml01"
STORAGE_ACCOUNT_KEY  = "YOUR_ACTUAL_KEY"
SILVER_CONTAINER     = "silver-layer"
SILVER_FILE          = "silver_icu_patients.parquet"

SUBSCRIPTION_ID      = "c7dbfd27-9bb9-4b43-b9e8-c7aa602333e6"
RESOURCE_GROUP       = "rg-healthcare-ml-dev"
WORKSPACE_NAME       = "healthcare-ml-workspace"
TENANT_ID            = "YOUR_ACTUAL_TENANT_ID"    # REPLACE
CLIENT_ID            = "YOUR_ACTUAL_CLIENT_ID"    # REPLACE
CLIENT_SECRET        = "YOUR_ACTUAL_SECRET"       # REPLACE

MODEL_NAME           = "sepsis-prediction-xgboost"
MODEL_VERSION        = "1"
TARGET_COL           = "sepsis_risk_label"

print("Config loaded.")


# ============================================================
# STEP 1 — Load Model from Azure ML Registry
# ============================================================
print("\nConnecting to Azure ML...")
credential = ClientSecretCredential(
    tenant_id     = TENANT_ID,
    client_id     = CLIENT_ID,
    client_secret = CLIENT_SECRET
)

ml_client = MLClient(
    credential          = credential,
    subscription_id     = SUBSCRIPTION_ID,
    resource_group_name = RESOURCE_GROUP,
    workspace_name      = WORKSPACE_NAME
)

print("Downloading model from Azure ML registry...")
model_info = ml_client.models.get(
    name    = MODEL_NAME,
    version = MODEL_VERSION
)

# Download model file
ml_client.models.download(
    name           = MODEL_NAME,
    version        = MODEL_VERSION,
    download_path  = "/tmp/sepsis_model"
)

# Load XGBoost model
model_path = f"/tmp/sepsis_model/{MODEL_NAME}/sepsis_xgboost_model.json"
model      = xgb.XGBClassifier()
model.load_model(model_path)
print(f"Model loaded: {MODEL_NAME} v{MODEL_VERSION}")

# ============================================================
# STEP 2 — Load Silver Data (Real 91K Patients)
# ============================================================
print("\nLoading silver data (real ICU patients)...")
connect_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT_NAME};"
    f"AccountKey={STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

blob_service   = BlobServiceClient.from_connection_string(connect_str)
silver_client  = blob_service.get_blob_client(
    container = SILVER_CONTAINER,
    blob      = SILVER_FILE
)

data = silver_client.download_blob().readall()
df   = pq.read_table(io.BytesIO(data)).to_pandas()
print(f"Loaded: {df.shape[0]:,} real ICU patients")

# ============================================================
# STEP 3 — Prepare Features
# ============================================================
exact_feature_order = [
    'temp_apache', 'gcs_verbal_apache',
    'd1_diasbp_min', 'heart_rate_apache',
    'hospital_death', 'h1_heartrate_min',
    'bmi', 'd1_heartrate_max',
    'pre_icu_los_days', 'd1_resprate_max',
    'd1_sysbp_min', 'd1_mbp_min',
    'map_apache', 'sepsis_risk_score',
    'd1_heartrate_min', 'gcs_motor_apache',
    'resprate_apache', 'ventilated_apache',
    'shock_index', 'pulse_pressure',
    'apache_4a_hospital_death_prob',
    'apache_4a_icu_death_prob', 'age',
    'h1_heartrate_max', 'h1_sysbp_min'
]

available   = [f for f in exact_feature_order if f in df.columns]
X_score     = df[available].copy()
X_score     = X_score.fillna(X_score.median())
X_score     = X_score[available]

print(f"Features prepared: {X_score.shape[1]} columns")
print(f"Order matches training: ")

# ============================================================
# STEP 4 — Generate Predictions
# ============================================================
print("\nGenerating sepsis risk predictions...")

risk_probabilities          = model.predict_proba(X_score)[:, 1]
binary_predictions          = model.predict(X_score)

df['sepsis_risk_score_pct'] = (risk_probabilities * 100).round(2)
df['sepsis_prediction']     = binary_predictions

def categorize_risk(score):
    if score >= 75:   return 'Critical'
    elif score >= 50: return 'High'
    elif score >= 25: return 'Medium'
    else:             return 'Low'

df['risk_category'] = df['sepsis_risk_score_pct'].apply(categorize_risk)

print(f"\nPredictions complete for {len(df):,} patients")
print("\nRisk Category Distribution:")
print(df['risk_category'].value_counts())
print(f"\nAverage sepsis risk score: {df['sepsis_risk_score_pct'].mean():.2f}%")

# ============================================================
# STEP 5 — Build Final Predictions Table
# ============================================================
output_cols = [
    'hospital_id', 'age', 'bmi',
    'shock_index', 'apache_4a_hospital_death_prob',
    'heart_rate_apache', 'temp_apache',
    'map_apache', 'resprate_apache',
    'hospital_death', 'pre_icu_los_days',
    'sepsis_risk_score_pct', 'sepsis_prediction',
    'risk_category', 'pulse_pressure',
    'ventilated_apache'
]

available_output     = [c for c in output_cols if c in df.columns]
df_predictions       = df[available_output].copy()
df_predictions['patient_id'] = range(1, len(df_predictions) + 1)

print(f"\nFinal predictions table: {df_predictions.shape}")
print(df_predictions[['patient_id', 'age',
    'sepsis_risk_score_pct', 'risk_category']].head(10))

# ============================================================
# STEP 6 — Write to Fabric Lakehouse
# ============================================================
print("\nWriting predictions to Fabric Lakehouse...")

spark_df = spark.createDataFrame(df_predictions)
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("patient_predictions")
print(" patient_predictions table written to Lakehouse")

# Summary table
summary_data = {
    'total_patients' : [len(df_predictions)],
    'critical_count' : [len(df_predictions[df_predictions['risk_category']=='Critical'])],
    'high_count'     : [len(df_predictions[df_predictions['risk_category']=='High'])],
    'medium_count'   : [len(df_predictions[df_predictions['risk_category']=='Medium'])],
    'low_count'      : [len(df_predictions[df_predictions['risk_category']=='Low'])],
    'avg_risk_score' : [round(df_predictions['sepsis_risk_score_pct'].mean(), 2)],
    'model_name'     : [MODEL_NAME],
    'model_version'  : [MODEL_VERSION],
}

df_summary    = pd.DataFrame(summary_data)
spark_summary = spark.createDataFrame(df_summary)
spark_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("prediction_summary")
print(" prediction_summary table written to Lakehouse")

# Feature importance table
importance_df = pd.DataFrame({
    'feature'    : available,
    'importance' : model.feature_importances_[:len(available)]
}).sort_values('importance', ascending=False)

spark_importance = spark.createDataFrame(importance_df)
spark_importance.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("feature_importance")
print(" feature_importance table written to Lakehouse")

print(f"""
╔══════════════════════════════════════════════╗
║         SCORING COMPLETE                     ║
╠══════════════════════════════════════════════╣
║  Total Patients Scored : {len(df_predictions):,}          ║
║  Critical Risk         : {len(df_predictions[df_predictions['risk_category']=='Critical']):,}           ║
║  High Risk             : {len(df_predictions[df_predictions['risk_category']=='High']):,}           ║
║  Medium Risk           : {len(df_predictions[df_predictions['risk_category']=='Medium']):,}          ║
║  Low Risk              : {len(df_predictions[df_predictions['risk_category']=='Low']):,}          ║
║  Avg Risk Score        : {df_predictions['sepsis_risk_score_pct'].mean():.2f}%              ║
║                                              ║
║  Tables in Lakehouse:                        ║
║   patient_predictions                      ║
║   prediction_summary                       ║
║   feature_importance                       ║
╚══════════════════════════════════════════════╝
""")


# In[ ]:




