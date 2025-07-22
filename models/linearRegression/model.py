import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(X_train, y_train, X_val, y_val):
    """
    Train a RidgeCV model. If X_val and y_val are provided, evaluates on validation set.
    """
    # Train RidgeCV model
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_train, y_train)

    # Evaluate if validation data is given
    if X_val is not None and y_val is not None:
        # Set max safe log value to avoid overflow in expm1
        MAX_LOG = 709  # np.log(np.finfo(np.float64).max)

        y_val_pred_log = ridge.predict(X_val)

        # Clip before applying expm1 to avoid overflow
        y_val_pred_log_clipped = np.clip(y_val_pred_log, a_min=None, a_max=MAX_LOG)
        y_val_clipped = np.clip(y_val, a_min=None, a_max=MAX_LOG)

        y_val_pred_real = np.expm1(y_val_pred_log_clipped)
        y_val_real = np.expm1(y_val_clipped)

        print("y_val_real contains NaNs:", np.isnan(y_val_real).any())
        print("y_val_real contains Infs:", np.isinf(y_val_real).any())
        print("y_val_pred_real contains NaNs:", np.isnan(y_val_pred_real).any())
        print("y_val_pred_real contains Infs:", np.isinf(y_val_pred_real).any())

        # Final metrics
        rmse = np.sqrt(mean_squared_error(y_val_real, y_val_pred_real))
        mae = mean_absolute_error(y_val_real, y_val_pred_real)
        r2 = r2_score(y_val_real, y_val_pred_real)

        print("\n[Validation Metrics]")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")

    return ridge



def predict_test(model, test_df, feature_cols, output_path="submission.csv"):
    test_ids = test_df["Id"]
    X_test = test_df.drop(columns=["Id", "SalePrice"], errors='ignore')
    X_test = X_test[feature_cols]  # select and order columns exactly

    # Handle missing values in test (fill with median or mean)
    if X_test.isnull().sum().sum() > 0:
        print(f"Warning: Test set contains missing values. Filling with median.")
        X_test = X_test.fillna(X_test.median())

    # Predict log prices
    y_test_pred_log = model.predict(X_test)

    # Clip to safe max before expm1 to avoid inf values
    MAX_LOG = 709  # approx max float64 log value to prevent overflow
    y_test_pred_log_clipped = np.clip(y_test_pred_log, a_min=None, a_max=MAX_LOG)

    # Transform back to original scale
    y_test_pred = np.expm1(y_test_pred_log_clipped)

    # Just in case, replace any remaining inf or NaN with a large finite number or median
    y_test_pred = np.where(np.isfinite(y_test_pred), y_test_pred, np.nan)
    if np.isnan(y_test_pred).any():
        median_pred = np.nanmedian(y_test_pred)
        y_test_pred = np.where(np.isnan(y_test_pred), median_pred, y_test_pred)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": y_test_pred
    })

    submission.to_csv(output_path, index=False)
    print(f"\n✅ Test predictions saved to {output_path}")
