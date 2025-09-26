import joblib
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Make sure data_utils.py is in the same folder or the path is correct
from data_utils import load_data, prepare_xy, train_val_split

# Function to build the machine learning pipeline
# This includes preprocessing and classifier

def build_pipeline():
    numeric_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop'
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return clf


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': float(accuracy_score(y_val, preds)),
        'precision': float(precision_score(y_val, preds, zero_division=0)),
        'recall': float(recall_score(y_val, preds, zero_division=0)),
        'f1': float(f1_score(y_val, preds, zero_division=0))
    }

    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_val, probs))
        except Exception:
            metrics['roc_auc'] = None

    return metrics


def main():
    # --- THIS IS THE KEY CHANGE ---
    # For local execution, provide the path to your data file.
    # The '..' tells Python to go up one directory from 'src' to find the CSV.
    data_path = "C:\\Users\\peeyu\\churn-prediction-project\\Customer churn train-test dataset\\customer_churn_dataset-training-master.csv"

    # Load and prepare data
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)

    # Build pipeline
    pipeline = build_pipeline()

    # Hyperparameter tuning grid (small scale)
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10]
    }

    print("Starting model training with GridSearchCV...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("Training complete.")

    # Evaluate on validation set
    metrics = evaluate_model(best_model, X_val, y_val)

    # Print best params and metrics
    print(f"\nBest params: {grid.best_params_}")
    print(f"Validation metrics: {metrics}")

    # Save model locally in a new 'model' directory
    output_dir = '../model'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'churn_rf.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()
