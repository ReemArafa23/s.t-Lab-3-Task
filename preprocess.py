import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess():
    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Split data
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save files
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    print("Preprocessing: Data split into train.csv and test.csv")

if __name__ == "__main__":
    preprocess()
