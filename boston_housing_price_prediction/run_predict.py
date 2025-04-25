import pandas as pd
from src import predict

# Load model and scaler
model, scaler = predict.load_artifacts()

# Input sample (edit this with real values you want to test)
sample = pd.DataFrame([{
    'CRIM': 0.25387,
    'ZN': 0.0,
    'INDUS': 6.91,
    'CHAS': 0.0,
    'NOX': 0.448,
    'RM': 5.339,
    'AGE': 95.3,
    'DIS': 5.87,
    'RAD': 3,
    'TAX': 233,
    'PTRATIO': 17.9,
    'B': 396.9,
    'LSTAT': 30.81
}])

# Get prediction
pred = predict.make_prediction(sample, model, scaler)
print("🏠 Predicted house price : ", round(pred[0], 2))
