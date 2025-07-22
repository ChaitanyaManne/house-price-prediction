import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_data(train):
    print("Dataset Info:\n")
    print(train.info())

    print("\nMissing Values:\n")
    missing = train.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

    print("\nStatistical Summary:\n")
    print(train.describe())

    print("\nTarget Variable Distribution:\n")
    sns.histplot(train['SalePrice'], kde=True)
    plt.title("Target variable distribution (SalePrice)")
    plt.show()

def get_feature_types(df):
    num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = df.select_dtypes(include=[object]).columns.tolist()

    print(f"\nNumerical features ({len(num_feats)}): {num_feats}")
    print(f"Categorical features ({len(cat_feats)}): {cat_feats}")
    return num_feats, cat_feats

def fill_missing_values(train, test, num_feats, cat_feats):
    for col in num_feats:
        if train[col].isnull().sum() > 0:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

    for col in cat_feats:
        if train[col].isnull().sum() > 0:
            mode_val = train[col].mode()[0]
            train[col] = train[col].fillna(mode_val)
            test[col] = test[col].fillna(mode_val)

    return train, test

def engineer_features(train, test):
    # Add new features
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

    train['Age'] = train['YrSold'] - train['YearBuilt']
    test['Age'] = test['YrSold'] - test['YearBuilt']

    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd']
    test['RemodAge'] = test['YrSold'] - test['YearRemodAdd']
    
    return train, test

def encode_and_merge_and_split(train, test, cat_feats):
    # Combine datasets for consistent encoding
    combined = pd.concat([train, test], sort=False).reset_index(drop=True)

    # One-hot encoding
    combined = pd.get_dummies(combined, columns=cat_feats, drop_first=True)

    # Split back
    train = combined.loc[:len(train)-1, :].copy()
    test = combined.loc[len(train):, :].copy()

    print(f"Train shape after encoding: {train.shape}")
    print(f"Test shape after encoding: {test.shape}")
    return train, test
