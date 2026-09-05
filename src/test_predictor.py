# =============================================================================
# TEST PREDICTOR
# =============================================================================

import sys
from pathlib import Path

import pandas as pd

# Add src directory to Python path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from predictor import predict_transaction


# =============================================================================
# LOAD DATASET
# =============================================================================

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("FRAUD DETECTION - PREDICTOR TEST")
print("=" * 70)

print("\nDataset shape:", df.shape)


# =============================================================================
# SELECT ONE NORMAL TRANSACTION
# =============================================================================

normal_transaction = df[
    df["Class"] == 0
].iloc[0]


# Remove Class because it is NOT an input feature
transaction_input = normal_transaction.drop(
    labels=["Class"]
).to_dict()


# =============================================================================
# PREDICTION
# =============================================================================

result = predict_transaction(
    transaction_input
)


# =============================================================================
# DISPLAY RESULT
# =============================================================================

print("\nPrediction Result")
print("-" * 70)

print(
    "Status:",
    result["status"]
)

print(
    "Reconstruction Error:",
    f"{result['reconstruction_error']:.8f}"
)

print(
    "Threshold:",
    f"{result['threshold']:.8f}"
)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "\nMessage:",
    result["message"]
)

print("\n✓ Predictor test completed successfully.")