# src/train.py
# Final version for model registration
import argparse
import os
import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import MLflow
import mlflow
import mlflow.sklearn

# --- Function to parse arguments ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--model_name", type=str, help="Name to register the model under")
    return parser.parse_args()

# --- Re-used functions from your original script ---
def load_data(path):
    df = pd.read_csv("C:\\Users\\peeyu\\churn-prediction-project\\Customer churn train-test dataset\\customer_churn_dataset-training-master.csv")
    return df

def prepare_xy(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# --- Main execution block ---
if __name__ == "__main__":
    args = parse_args()

    # --- 1. Load Data ---
    print("Loading data...")
    full_df = load_data(args.data_path)
    X, y = prepare_xy(full_df, target_col='Churn')

    # --- 2. Build Preprocessing Pipeline ---
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 3. Build Training Pipeline ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')

    # --- THIS IS THE CRITICAL NEW PART: Using MLflow ---
    # Start MLflow logging
    mlflow.autolog()
    
    print("Starting model training with GridSearchCV...")
    grid_search.fit(X, y)
    
    print("Training complete.")
    best_model = grid_search.best_estimator_

    # --- 4. Register the Model with MLflow ---
    # The 'autolog' feature handles logging metrics and parameters automatically.
    # Now we explicitly register the best model found by the grid search.
    print(f"Registering model as: {args.model_name}")
    
    # This line tells Azure ML to take the trained model and save it in the Model Registry
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model", # This is a folder name within the run's artifacts
        registered_model_name=args.model_name # This is the name the deployment step will look for
    )

    print("Model registered successfully!")

