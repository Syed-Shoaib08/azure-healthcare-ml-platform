#!/usr/bin/env python
# coding: utf-8

# ## 03_train_model
# 
# null

# In[3]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install azure-ai-ml azure-identity xgboost scikit-learn pyarrow azure-storage-blob


# In[4]:


# ============================================================
# 03_train_model.py
# Healthcare ML Platform — XGBoost Sepsis Prediction Model
# ============================================================

import pandas as pd
import numpy as np
import io
import joblib
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import xgboost as xgb
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# ============================================================
# CONFIGURATION
# ============================================================
STORAGE_ACCOUNT_NAME = "sthealthcareml01"
STORAGE_ACCOUNT_KEY  = "YOUR_ACTUAL_KEY"
GOLD_CONTAINER       = "gold-layer"
TARGET_COL           = "sepsis_risk_label"

SUBSCRIPTION_ID      = "YOUR_SUBSCRIPTION_ID"
RESOURCE_GROUP       = "rg-healthcare-ml-dev"
WORKSPACE_NAME       = "healthcare-ml-workspace"

print("Config loaded.")

# ============================================================
# STEP 1 — Connect to Azure Storage + Load Gold Data
# ============================================================
connect_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT_NAME};"
    f"AccountKey={STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

blob_service = BlobServiceClient.from_connection_string(connect_str)
print("Connected to Azure Storage.")

# Read all 5 gold partitions
print("Loading gold data (20M rows — this takes a few minutes)...")
dfs = []
for i in range(1, 6):
    filename    = f"gold_ml_ready_part{i}.parquet"
    client      = blob_service.get_blob_client(
        container=GOLD_CONTAINER,
        blob=filename
    )
    data        = client.download_blob().readall()
    df_part     = pq.read_table(io.BytesIO(data)).to_pandas()
    dfs.append(df_part)
    print(f"  Part {i}/5 loaded — {len(df_part):,} rows")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal gold data: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2 — Prepare Features and Target
# ============================================================
y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL, 'record_id'], errors='ignore')

# Fill any nulls
X = X.fillna(X.median())

print(f"\nFeatures: {X.shape[1]} columns")
print(f"Target distribution:\n{y.value_counts()}")

# ============================================================
# STEP 3 — Train/Test Split (80/20)
# ============================================================
print("\nSplitting data 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]:,} rows")
print(f"Testing set:  {X_test.shape[0]:,} rows")

# ============================================================
# STEP 4 — Train XGBoost Model
# ============================================================
print("\nTraining XGBoost model...")
print("(Training on 16M rows — expect 10-15 minutes)")

model = xgb.XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    use_label_encoder = False,
    eval_metric       = 'logloss',
    random_state      = 42,
    n_jobs            = -1,
    tree_method       = 'hist'  # fastest method for large data
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("\nModel training complete!")

# ============================================================
# STEP 5 — Evaluate Model
# ============================================================
print("\nEvaluating model on test set (4M rows)...")
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

precision   = precision_score(y_test, y_pred)
recall      = recall_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
auc_roc     = roc_auc_score(y_test, y_pred_prob)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"F1 Score  : {f1:.4f}  ({f1*100:.2f}%)")
print(f"AUC-ROC   : {auc_roc:.4f}  ({auc_roc*100:.2f}%)")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
    target_names=['No Sepsis', 'Sepsis']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives  (correctly said No Sepsis): {cm[0][0]:,}")
print(f"False Positives (wrongly said Sepsis):      {cm[0][1]:,}")
print(f"False Negatives (missed Sepsis):            {cm[1][0]:,}")
print(f"True Positives  (correctly said Sepsis):    {cm[1][1]:,}")

# Feature importance
print("\nTop 10 Most Important Features:")
importance_df = pd.DataFrame({
    'feature'   : X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(10).to_string(index=False))

# ============================================================
# STEP 6 — Save Model + Register in Azure ML
# ============================================================
print("\nSaving model...")
model_filename = "sepsis_xgboost_model.json"
model.save_model(model_filename)
print(f"Model saved as {model_filename}")

# Connect to Azure ML using Service Principal
print("\nConnecting to Azure ML workspace...")
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Model as AzureModel
from azure.ai.ml.constants import AssetTypes

TENANT_ID            = "YOUR_ACTUAL_TENANT_ID"    # REPLACE
CLIENT_ID            = "YOUR_ACTUAL_CLIENT_ID"    # REPLACE
CLIENT_SECRET        = "YOUR_ACTUAL_SECRET"       # REPLACE

credential = ClientSecretCredential(
    tenant_id     = TENANT_ID,
    client_id     = CLIENT_ID,
    client_secret = CLIENT_SECRET
)

ml_client = MLClient(
    credential      = credential,
    subscription_id = "c7dbfd27-9bb9-4b43-b9e8-c7aa602333e6",
    resource_group_name  = "rg-healthcare-ml-dev",
    workspace_name  = WORKSPACE_NAME
)

print(f"Connected to workspace: {WORKSPACE_NAME}")

# Register model
print("Registering model in Azure ML...")
azure_model = AzureModel(
    path        = model_filename,
    name        = "sepsis-prediction-xgboost",
    description = "XGBoost sepsis prediction model trained on 20M ICU patient records",
    type        = AssetTypes.CUSTOM_MODEL,
    tags        = {
        "algorithm" : "XGBoost",
        "dataset"   : "Kaggle ICU + SMOTE 20M",
        "f1_score"  : str(round(f1, 4)),
        "auc_roc"   : str(round(auc_roc, 4)),
        "framework" : "xgboost"
    }
)

registered = ml_client.models.create_or_update(azure_model)

print(f"\n Model registered successfully!")
print(f"   Name   : {registered.name}")
print(f"   Version: {registered.version}")
print(f"\n Training pipeline complete.")


# In[ ]:




