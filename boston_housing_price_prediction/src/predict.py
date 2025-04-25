import joblib
import pandas as pd

def load_artifacts(model_path='models/xgb_model.pkl', scaler_path='models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def make_prediction(input_data: pd.DataFrame, model, scaler):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction
