# =============================================================================
# predictor.py
# Credit Card Fraud Detection - Autoencoder Inference
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root:
# Fraud Detection/
# ├── models/
# └── src/
#     └── predictor.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


AUTOENCODER_PATH = MODEL_DIR / "autoencoder.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"


# =============================================================================
# LOAD TRAINED MODEL AND SUPPORTING FILES
# =============================================================================

def load_model_artifacts():
    """
    Load the trained Autoencoder, scaler, and anomaly threshold.
    """

    if not AUTOENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Autoencoder model not found: {AUTOENCODER_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    autoencoder = tf.keras.models.load_model(
        AUTOENCODER_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    threshold_data = joblib.load(
        THRESHOLD_PATH
    )

    # threshold.pkl contains a dictionary created during training
    if isinstance(threshold_data, dict):
        threshold = float(
            threshold_data["threshold"]
        )
    else:
        threshold = float(threshold_data)

    return autoencoder, scaler, threshold


# =============================================================================
# LOAD ARTIFACTS ONCE
# =============================================================================

autoencoder, scaler, THRESHOLD = load_model_artifacts()


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# =============================================================================
# RECONSTRUCTION ERROR
# =============================================================================

def calculate_reconstruction_error(
    original_data,
    reconstructed_data
):
    """
    Calculate Mean Squared Reconstruction Error
    for each transaction.
    """

    errors = np.mean(
        np.square(
            original_data - reconstructed_data
        ),
        axis=1
    )

    return errors


# =============================================================================
# PREDICT TRANSACTION
# =============================================================================

def predict_transaction(transaction):
    """
    Predict whether a credit card transaction is
    normal or anomalous.

    Parameters
    ----------
    transaction : dict or pandas.DataFrame
        Must contain the 30 model input features:
        Time, V1-V28, Amount.

    Returns
    -------
    dict
        Prediction result containing:
        - prediction
        - reconstruction_error
        - threshold
        - status
    """

    # -------------------------------------------------------------------------
    # Convert input to DataFrame
    # -------------------------------------------------------------------------

    if isinstance(transaction, dict):

        transaction_df = pd.DataFrame(
            [transaction]
        )

    elif isinstance(transaction, pd.DataFrame):

        transaction_df = transaction.copy()

    else:

        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )


    # -------------------------------------------------------------------------
    # Validate required features
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in transaction_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )


    # -------------------------------------------------------------------------
    # Keep features in EXACT training order
    # -------------------------------------------------------------------------

    transaction_df = transaction_df[
        FEATURES
    ]


    # -------------------------------------------------------------------------
    # Validate numeric values
    # -------------------------------------------------------------------------

    if transaction_df.isnull().any().any():

        raise ValueError(
            "Input contains missing values."
        )

    try:

        transaction_df = transaction_df.astype(
            float
        )

    except ValueError as error:

        raise ValueError(
            "All transaction features must be numeric."
        ) from error


    # -------------------------------------------------------------------------
    # Scale using the SAME scaler used during training
    # -------------------------------------------------------------------------

    scaled_data = scaler.transform(
        transaction_df
    )


    # -------------------------------------------------------------------------
    # Reconstruct transaction
    # -------------------------------------------------------------------------

    reconstructed_data = autoencoder.predict(
        scaled_data,
        verbose=0
    )


    # -------------------------------------------------------------------------
    # Calculate reconstruction error
    # -------------------------------------------------------------------------

    reconstruction_errors = calculate_reconstruction_error(
        scaled_data,
        reconstructed_data
    )

    reconstruction_error = float(
        reconstruction_errors[0]
    )


    # -------------------------------------------------------------------------
    # Apply trained threshold
    # -------------------------------------------------------------------------

    is_anomaly = (
        reconstruction_error > THRESHOLD
    )


    # -------------------------------------------------------------------------
    # Prepare result
    # -------------------------------------------------------------------------

    if is_anomaly:

        prediction = 1
        status = "Fraud / Anomaly"
        message = (
            "Transaction flagged as anomalous "
            "because its reconstruction error "
            "exceeds the trained threshold."
        )

    else:

        prediction = 0
        status = "Normal"
        message = (
            "Transaction classified as normal "
            "because its reconstruction error "
            "is within the normal range."
        )


    return {
        "prediction": prediction,
        "status": status,
        "reconstruction_error": reconstruction_error,
        "threshold": THRESHOLD,
        "message": message
    }


# =============================================================================
# BATCH PREDICTION
# =============================================================================

def predict_transactions(transactions):
    """
    Predict multiple transactions at once.

    Parameters
    ----------
    transactions : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Original transaction data with:
        - Reconstruction_Error
        - Prediction
        - Status
    """

    if not isinstance(
        transactions,
        pd.DataFrame
    ):

        raise TypeError(
            "Batch input must be a pandas DataFrame."
        )


    # -------------------------------------------------------------------------
    # Validate required features
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in transactions.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )


    # -------------------------------------------------------------------------
    # Select features in training order
    # -------------------------------------------------------------------------

    input_data = transactions[
        FEATURES
    ].copy()


    # -------------------------------------------------------------------------
    # Validate data
    # -------------------------------------------------------------------------

    if input_data.isnull().any().any():

        raise ValueError(
            "Input contains missing values."
        )


    # -------------------------------------------------------------------------
    # Convert to numeric
    # -------------------------------------------------------------------------

    try:

        input_data = input_data.astype(
            float
        )

    except ValueError as error:

        raise ValueError(
            "All transaction features must be numeric."
        ) from error


    # -------------------------------------------------------------------------
    # Scale input
    # -------------------------------------------------------------------------

    scaled_data = scaler.transform(
        input_data
    )


    # -------------------------------------------------------------------------
    # Reconstruct transactions
    # -------------------------------------------------------------------------

    reconstructed_data = autoencoder.predict(
        scaled_data,
        verbose=0
    )


    # -------------------------------------------------------------------------
    # Calculate reconstruction errors
    # -------------------------------------------------------------------------

    reconstruction_errors = calculate_reconstruction_error(
        scaled_data,
        reconstructed_data
    )


    # -------------------------------------------------------------------------
    # Generate predictions
    # -------------------------------------------------------------------------

    predictions = (
        reconstruction_errors > THRESHOLD
    ).astype(int)


    # -------------------------------------------------------------------------
    # Create output
    # -------------------------------------------------------------------------

    results = transactions.copy()

    results["Reconstruction_Error"] = (
        reconstruction_errors
    )

    results["Prediction"] = predictions

    results["Status"] = np.where(
        predictions == 1,
        "Fraud / Anomaly",
        "Normal"
    )

    return results


# =============================================================================
# MODEL INFORMATION
# =============================================================================

def get_model_info():
    """
    Return information about the loaded model.
    """

    return {
        "model": "Credit Card Fraud Autoencoder",
        "input_features": len(FEATURES),
        "features": FEATURES,
        "threshold": THRESHOLD,
        "threshold_method": (
            "99th percentile of normal validation "
            "reconstruction errors"
        )
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - MODEL TEST")
    print("=" * 70)

    print("\nModel loaded successfully.")
    print(
        "Number of input features:",
        len(FEATURES)
    )

    print(
        "Anomaly threshold:",
        f"{THRESHOLD:.8f}"
    )

    print("\nRequired features:")
    print(FEATURES)

    print("\n✓ predictor.py is working correctly.")