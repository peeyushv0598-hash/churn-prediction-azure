# In src/data_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def prepare_xy(df):
    """
    Cleans the dataframe by dropping rows with NaN in the 'Churn' column,
    and then separates features (X) and target (y).
    """
    df = df.copy()

    # --- START OF FIX ---
    # Drop rows where the 'Churn' column is NaN before doing anything else
    df.dropna(subset=['Churn'], inplace=True)
    # --- END OF FIX ---

    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    
    # Ensure Churn is integer type after dropping NaNs
    df['Churn'] = df['Churn'].astype(int)

    # Separate target
    y = df['Churn']
    X = df.drop(columns=['Churn'])

    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
