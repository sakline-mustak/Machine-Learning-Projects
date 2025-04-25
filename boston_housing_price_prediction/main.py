from src import data_loader, preprocess, train_model, evaluate
import joblib

def main():
    df = data_loader.load_data('data/raw/Boston House Price Data.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess.preprocess(df)
    
    model = train_model.train_model(X_train, y_train)
    metrics = evaluate.evaluate(model, X_test, y_test)
    print("Evaluation:", metrics)
    
    joblib.dump(model, 'models/xgb_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == "__main__":
    main()
