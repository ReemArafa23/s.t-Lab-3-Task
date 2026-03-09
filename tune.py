import pandas as pd
import mlflow
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def tune_model():
    # Load data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    X_train, y_train = train_data.drop('quality', axis=1), train_data['quality']
    X_test, y_test = test_data.drop('quality', axis=1), test_data['quality']

    # Parent Run for Tuning
    with mlflow.start_run(run_name="SVM_Hyperparameter_Tuning"):
        # Test two different C values (Hyperparameters)
        for c_value in [0.1, 1.0]:
            with mlflow.start_run(run_name=f"SVM_C_{c_value}", nested=True):
                model = SVC(C=c_value, kernel='rbf')
                model.fit(X_train, y_train)
                
                acc = accuracy_score(y_test, model.predict(X_test))
                
                mlflow.log_param("C", c_value)
                mlflow.log_metric("accuracy", acc)
                print(f"Tuning: Nested run C={c_value} completed with Accuracy={acc}")

if __name__ == "__main__":
    tune_model()
