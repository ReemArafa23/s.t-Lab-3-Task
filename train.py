import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load training data
    train_data = pd.read_csv('data/train.csv')
    X_train = train_data.drop('quality', axis=1)
    y_train = train_data['quality']

    # Start MLflow run
    with mlflow.start_run(run_name="Baseline_RandomForest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model locally for DVC
        joblib.dump(model, 'model.joblib')
        
        # Log to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Training: Baseline model saved and logged to MLflow.")

if __name__ == "__main__":
    train_model()
