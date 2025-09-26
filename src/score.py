import os
import json
import joblib
import pandas as pd

# This global variable will hold the model after it's loaded
model = None

def init():
    """
    This function is called when the container for the endpoint is initialized.
    It loads the model from the path where Azure ML places it.
    """
    global model
    
    # AZUREML_MODEL_DIR is an environment variable created by Azure.
    # It points to the directory where the model files are located.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/model.pkl") # MLflow default path
    
    # Add a fallback for flexibility
    if not os.path.exists(model_path):
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "churn_rf.joblib")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded.")


def run(raw_data):
    """
    This function is called for each prediction request to the endpoint.
    """
    try:
        # The input is a raw JSON string. We need to parse it.
        data = json.loads(raw_data)
        
        # Your predict_single logic is used here: convert dict to DataFrame
        df = pd.DataFrame([data])
        
        # Make the prediction
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1] if hasattr(model, "predict_proba") else None
        
        # Return the result as a dictionary, which Azure will convert to JSON
        return {
            'prediction': int(pred), 
            'probability': float(proba) if proba is not None else None
        }

    except Exception as e:
        # Return an error message if something goes wrong
        return {"error": str(e)}

