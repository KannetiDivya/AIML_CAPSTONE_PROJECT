# =============================================================================
# CREDIT CARD FRAUD DETECTION - FRAUDGUARD
# Streamlit Application
# =============================================================================

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="FraudGuard | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "creditcard.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

AUTOENCODER_PATH = os.path.join(
    MODELS_DIR,
    "autoencoder.keras"
)

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "threshold.pkl"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    /* ============================================================
       GLOBAL
       ============================================================ */

    .stApp {
        background:
            radial-gradient(
                circle at 90% 5%,
                rgba(37, 99, 235, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 5% 90%,
                rgba(14, 165, 233, 0.08),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f5f8fc 0%,
                #eef3f9 50%,
                #f8fafc 100%
            );
    }

    .main {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* ============================================================
       SIDEBAR
       ============================================================ */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #071426 0%,
                #0b1d35 55%,
                #102a48 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #eaf2ff !important;
    }

    .brand-box {
        padding: 8px 4px 24px 4px;
    }

    .brand-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: white !important;
        margin-bottom: 3px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #a9bdd5 !important;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 700;
        color: #8ea8c5 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    /* ============================================================
       HEADINGS
       ============================================================ */

    .page-title {
        font-size: 38px;
        font-weight: 800;
        color: #0b1f3a;
        letter-spacing: -1.2px;
        margin-bottom: 4px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #60738c;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #102744;
        margin-top: 25px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 14px;
        color: #6b7d93;
        margin-bottom: 18px;
    }

    /* ============================================================
       CARDS
       ============================================================ */

    .metric-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #dce5ef;
        border-radius: 18px;
        padding: 24px;
        min-height: 145px;
        box-shadow:
            0 8px 25px rgba(15, 35, 60, 0.06);
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 30px rgba(15, 35, 60, 0.09);
    }

    .metric-label {
        font-size: 13px;
        color: #64778e;
        margin-bottom: 13px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #0a2340;
    }

    .info-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #dce5ef;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        box-shadow:
            0 8px 25px rgba(15, 35, 60, 0.05);
    }

    .info-card-title {
        font-size: 18px;
        font-weight: 750;
        color: #102744;
        margin-bottom: 12px;
    }

    .info-card-text {
        color: #52677f;
        line-height: 1.8;
        font-size: 14px;
    }

    /* ============================================================
       RESULT CARDS
       ============================================================ */

    .result-normal {
        background: linear-gradient(
            135deg,
            #ecfdf5,
            #f5fffb
        );
        border: 1px solid #86efac;
        border-left: 5px solid #16a34a;
        border-radius: 16px;
        padding: 24px;
        margin-top: 22px;
        box-shadow: 0 8px 25px rgba(22,163,74,0.08);
    }

    .result-fraud {
        background: linear-gradient(
            135deg,
            #fff1f2,
            #fff8f8
        );
        border: 1px solid #fca5a5;
        border-left: 5px solid #dc2626;
        border-radius: 16px;
        padding: 24px;
        margin-top: 22px;
        box-shadow: 0 8px 25px rgba(220,38,38,0.08);
    }

    .result-title {
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .result-normal .result-title {
        color: #166534;
    }

    .result-fraud .result-title {
        color: #991b1b;
    }

    .result-text {
        font-size: 14px;
        line-height: 1.7;
        color: #52677f;
    }

    /* ============================================================
       BUTTONS
       ============================================================ */

    .stButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        color: white;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 5px 15px rgba(37,99,235,0.20);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #1d4ed8,
            #1e40af
        );
        transform: translateY(-1px);
    }

    /* ============================================================
       INPUTS
       ============================================================ */

    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    div[data-baseweb="input"] > div {
        border-radius: 10px;
    }

    /* ============================================================
       DATAFRAME
       ============================================================ */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #dce5ef;
    }

    /* ============================================================
       FOOTER
       ============================================================ */

    .footer {
        text-align: center;
        color: #8292a6;
        font-size: 11px;
        margin-top: 45px;
        padding: 15px;
    }

    /* ============================================================
       RESPONSIVE
       ============================================================ */

    @media (max-width: 900px) {

        .page-title {
            font-size: 30px;
        }

        .page-subtitle {
            font-size: 14px;
        }

        .metric-card {
            min-height: 125px;
            padding: 20px;
        }

        .metric-value {
            font-size: 24px;
        }
    }

    @media (max-width: 600px) {

        .main {
            padding-left: 8px;
            padding-right: 8px;
        }

        .page-title {
            font-size: 26px;
        }

        .section-title {
            font-size: 20px;
        }

        .metric-card {
            padding: 18px;
        }

        .metric-value {
            font-size: 22px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# LOAD MODEL FILES
# =============================================================================

@st.cache_resource
def load_model_files():

    if not os.path.exists(AUTOENCODER_PATH):
        raise FileNotFoundError(
            f"Autoencoder model not found:\n{AUTOENCODER_PATH}"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}"
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_PATH}"
        )

    autoencoder = tf.keras.models.load_model(
        AUTOENCODER_PATH,
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "rb") as f:
        threshold_data = pickle.load(f)

    if isinstance(threshold_data, dict):
        threshold = float(threshold_data["threshold"])
    else:
        threshold = float(threshold_data)

    return autoencoder, scaler, threshold


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_test_predictions():

    if not os.path.exists(PREDICTIONS_PATH):
        return None

    return pd.read_csv(PREDICTIONS_PATH)


# =============================================================================
# SCALE INPUT DATA
# =============================================================================

def prepare_input(features, scaler):

    features = np.asarray(features, dtype=np.float32)

    try:

        # If scaler was trained on all 30 features
        if hasattr(scaler, "n_features_in_"):

            if scaler.n_features_in_ == 30:

                return scaler.transform(
                    features.reshape(1, -1)
                )

            # If scaler was trained only on Time and Amount
            elif scaler.n_features_in_ == 2:

                scaled = features.copy()

                scaled[0, 0:2] = scaler.transform(
                    features[:, 0:2]
                )

                return scaled

        # Fallback: scale Time and Amount
        scaled = features.copy()

        scaled[0, 0:2] = scaler.transform(
            features[:, 0:2]
        )

        return scaled

    except Exception as e:

        raise RuntimeError(
            f"Input scaling failed: {str(e)}"
        )


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_transaction(features):

    autoencoder, scaler, threshold = load_model_files()

    X = prepare_input(features, scaler)

    reconstructed = autoencoder.predict(
        X,
        verbose=0
    )

    reconstruction_error = float(
        np.mean(
            np.square(
                X - reconstructed
            )
        )
    )

    prediction = int(
        reconstruction_error > threshold
    )

    return (
        prediction,
        reconstruction_error,
        threshold
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">🛡️ FraudGuard</div>
            <div class="brand-subtitle">
                Credit Card Fraud Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Menu</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Select Section",
        [
            "Fraud Analytics",
            "Transaction Investigation",
            "New Transaction Screening"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#8ea8c5;
            font-size:12px;
            line-height:1.6;
        ">
        Autoencoder-Based
        Anomaly Detection
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# DATA
# =============================================================================

df = load_dataset()
predictions_df = load_test_predictions()


# =============================================================================
# PAGE 1 - FRAUD ANALYTICS
# =============================================================================

if page == "Fraud Analytics":

    st.markdown(
        '<div class="page-title">Credit Card Fraud Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-subtitle">
            Autoencoder-based anomaly detection for identifying
            suspicious credit card transactions.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Fraud Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Project overview and trained model information.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # MODEL INFORMATION
    # -------------------------------------------------------------------------

    try:
        _, _, threshold = load_model_files()
    except Exception:
        threshold = 0.21305744

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Detection Model
                </div>
                <div class="metric-value">
                    Autoencoder
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Input Features
                </div>
                <div class="metric-value">
                    30
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Detection Threshold
                </div>
                <div class="metric-value">
                    {threshold:.8f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # PROJECT STATUS
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Project Status</div>',
        unsafe_allow_html=True
    )

    if df is not None:

        total_transactions = len(df)

        if "Class" in df.columns:
            fraud_cases = int(
                df["Class"].sum()
            )
        else:
            fraud_cases = 492

        normal_cases = total_transactions - fraud_cases

    else:

        total_transactions = 284807
        normal_cases = 284315
        fraud_cases = 492

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Dataset Transactions
                </div>
                <div class="metric-value">
                    {total_transactions:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Normal Transactions
                </div>
                <div class="metric-value">
                    {normal_cases:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Known Fraud Cases
                </div>
                <div class="metric-value">
                    {fraud_cases:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# PAGE 2 - TRANSACTION INVESTIGATION
# =============================================================================

elif page == "Transaction Investigation":

    st.markdown(
        '<div class="page-title">Transaction Investigation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-subtitle">
            Investigate an existing transaction from the evaluation dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # INVESTIGATE TRANSACTION
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Investigate Transaction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Enter a transaction index to view its anomaly detection result.
        </div>
        """,
        unsafe_allow_html=True
    )

    if predictions_df is not None:

        max_index = len(predictions_df) - 1

        transaction_index = st.number_input(
            "Transaction Index",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1
        )

        if st.button(
            "Investigate Transaction",
            use_container_width=True
        ):

            row = predictions_df.iloc[
                int(transaction_index)
            ]

            prediction = int(
                row["Predicted_Anomaly"]
            )

            error = float(
                row["Reconstruction_Error"]
            )

            if "Threshold" in row.index:
                threshold = float(row["Threshold"])
            else:
                threshold = float(load_model_files()[2])

            if prediction == 1:

                st.html(
                    f"""
<div class="result-fraud">

    <div class="result-title">
        ⚠️ Fraud / Anomaly Detected
    </div>

    <div class="result-text">
        Transaction <b>#{int(transaction_index)}</b>
        has been flagged because its reconstruction
        error exceeds the anomaly threshold.
        <br><br>
        <b>Reconstruction Error:</b>
        {error:.8f}
        <br>
        <b>Threshold:</b>
        {threshold:.8f}
    </div>

</div>
"""
                )

            else:

                st.html(
                    f"""
<div class="result-normal">

    <div class="result-title">
        ✓ Normal Transaction
    </div>

    <div class="result-text">
        Transaction <b>#{int(transaction_index)}</b>
        is within the learned normal range.
        <br><br>
        <b>Reconstruction Error:</b>
        {error:.8f}
        <br>
        <b>Threshold:</b>
        {threshold:.8f}
    </div>

</div>
"""
                )

    else:

        st.error(
            "test_predictions.csv was not found in the results folder."
        )

    # -------------------------------------------------------------------------
    # FLAGGED TRANSACTIONS
    # -------------------------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Flagged Transactions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Transactions identified as anomalies by the Autoencoder.
        </div>
        """,
        unsafe_allow_html=True
    )

    if predictions_df is not None:

        flagged_df = predictions_df[
            predictions_df["Predicted_Anomaly"] == 1
        ].copy()

        display_columns = [
            column
            for column in [
                "Transaction_Index",
                "Time",
                "Amount",
                "Reconstruction_Error",
                "Detection_Status",
                "Reason"
            ]
            if column in flagged_df.columns
        ]

        if len(flagged_df) > 0:

            display_df = flagged_df[
                display_columns
            ].head(10)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No flagged transactions available."
            )

    else:

        st.warning(
            "Flagged transaction results are unavailable."
        )


# =============================================================================
# PAGE 3 - NEW TRANSACTION SCREENING
# =============================================================================

elif page == "New Transaction Screening":

    st.markdown(
        '<div class="page-title">New Transaction Screening</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-subtitle">
            Screen a transaction using the trained Autoencoder model.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # TRANSACTION INFORMATION
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Transaction Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Enter the transaction time and amount.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        transaction_time = st.number_input(
            "Time",
            value=0.0,
            format="%.6f"
        )

    with col2:

        transaction_amount = st.number_input(
            "Amount",
            value=0.0,
            min_value=0.0,
            format="%.6f"
        )

    # -------------------------------------------------------------------------
    # 28 PCA FEATURES
    # -------------------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Transaction Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Enter the 28 anonymized PCA transaction features.
        </div>
        """,
        unsafe_allow_html=True
    )

    feature_values = []

    columns = st.columns(4)

    for i in range(28):

        with columns[i % 4]:

            value = st.number_input(
                f"V{i + 1}",
                value=0.0,
                format="%.6f",
                key=f"v{i + 1}"
            )

            feature_values.append(value)

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    if st.button(
        "Analyze Transaction",
        use_container_width=True
    ):

        try:

            features = [
                transaction_time,
                transaction_amount
            ] + feature_values

            prediction, reconstruction_error, threshold = (
                predict_transaction(features)
            )

            if prediction == 1:

                st.html(
                    f"""
<div class="result-fraud">

    <div class="result-title">
        ⚠️ Fraud / Anomaly Detected
    </div>

    <div class="result-text">
        The transaction has been flagged because
        its reconstruction error exceeds the
        anomaly threshold.

        <br><br>

        <b>Reconstruction Error:</b>
        {reconstruction_error:.8f}

        <br>

        <b>Detection Threshold:</b>
        {threshold:.8f}

    </div>

</div>
"""
                )

            else:

                st.html(
                    f"""
<div class="result-normal">

    <div class="result-title">
        ✓ Normal Transaction
    </div>

    <div class="result-text">
        The transaction is within the learned
        normal range.

        <br><br>

        <b>Reconstruction Error:</b>
        {reconstruction_error:.8f}

        <br>

        <b>Detection Threshold:</b>
        {threshold:.8f}

    </div>

</div>
"""
                )

        except Exception as e:

            st.error(
                f"Prediction could not be completed: {str(e)}"
            )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div class="footer">
        Credit Card Fraud Detection • Autoencoder-Based Anomaly Detection
    </div>
    """,
    unsafe_allow_html=True
)