import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    print("="*60)
    print(" UFCFAS-15-2 Machine Learning: Containerized Model Training")
    print("="*60)
    
    data_file = "data.csv"
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)
        
    print(f"\nLoading dataset from '{data_file}'...")
    df = pd.read_csv(data_file)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nSample records:")
    print(df.head(5).to_string(index=False))
    
    # Feature matrix X and target vector y
    X = df[["study_hours"]].values
    y = df["exam_score"].values
    
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("\nModel Training Complete!")
    print(f"Learned Equation: Exam Score = ({slope:.2f} * Study Hours) + {intercept:.2f}")
    print(f"Slope (Weight w): {slope:.4f}")
    print(f"Intercept (Bias b): {intercept:.4f}")
    print(f"R2 Score (Accuracy): {r2:.4f} ({r2*100:.1f}%)")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Run test predictions
    print("\nInference Test (New Predictions):")
    test_hours = np.array([[3.0], [5.0], [7.5], [10.0]])
    predictions = model.predict(test_hours)
    for hours, pred in zip(test_hours.ravel(), predictions):
        print(f"Studying {hours:4.1f} hours/week -> Predicted Score: {pred:5.1f} / 100")
        
    # Save model artifact
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to: '{model_path}'")
    print("="*60)
    print("Container run successfully finished!")
    print("="*60)

if __name__ == "__main__":
    main()