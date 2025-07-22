from load_data import load_data
from preprocessing import summarize_data
from preprocessing import get_feature_types
from preprocessing import fill_missing_values, engineer_features, encode_and_merge_and_split
from model import train_model, predict_test
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # Step 1: Load raw train and test data
    train_raw, test_raw = load_data("../../data/train.csv", "../../data/test.csv")
    print("✅ Data loaded.")

    # Step 2: Preprocess both train and test sets
    summarize_data(train_raw)   
    num_feats, cat_feats = get_feature_types(train_raw)
    train, test = fill_missing_values(train_raw, test_raw, num_feats, cat_feats)
    train, test = engineer_features(train, test)
    train, test = encode_and_merge_and_split(train, test, cat_feats)
    
    


    train , val = train_test_split(train, test_size = 0.15, random_state = 1234)
    X_train = train.drop(columns=["SalePrice", "Id"])
    y_train = np.log1p(train["SalePrice"])
    X_val = val.drop(columns=["SalePrice", "Id"])
    y_val = np.log1p(val["SalePrice"]) 
    feature_cols = X_train.columns.tolist()
    print("✅ Data preprocessing completed.")

    # Step 3: Train model and evaluate on validation split
    model = train_model(X_train, y_train, X_val, y_val)
    print("✅ Model trained.")

    # Step 4: Predict on test set and save submission.cs
    predict_test(model, test, feature_cols=feature_cols, output_path="submission.csv")
    print("✅ Submission file created.")

if __name__ == "__main__":
    main()
