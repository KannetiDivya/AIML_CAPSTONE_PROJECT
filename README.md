# 🛡️ FraudGuard — Fraud Detection Using Autoencoders

An unsupervised machine learning project for detecting unusual credit card transactions using an Autoencoder-based anomaly detection approach.

---

## 📌 Project Overview

Credit card fraud detection is challenging because fraudulent transactions represent only a very small proportion of total transactions.

This project uses an **Autoencoder-based anomaly detection approach** to learn the patterns of normal credit card transactions.

The Autoencoder is trained using **normal transactions only**. Fraud labels are not used during model training.

After training, each transaction is evaluated using its **reconstruction error**. Transactions with reconstruction errors above the selected threshold are flagged as anomalous and can be investigated further.

The project also compares the Autoencoder with **Isolation Forest** and **One-Class SVM**.

A Streamlit web application called **FraudGuard** is provided for transaction investigation and new transaction screening.

---

## 🎯 Project Objective

The main objectives are to:

- Detect unusual credit card transactions using an unsupervised approach.
- Train an Autoencoder using normal transactions only.
- Calculate reconstruction error for each transaction.
- Select an anomaly threshold using normal validation data.
- Evaluate the model on a held-out test dataset.
- Compare the Autoencoder with Isolation Forest and One-Class SVM.
- Provide a Streamlit application for transaction screening and investigation.

---

## 📊 Dataset

This project uses the **ULB Machine Learning Group Credit Card Fraud Detection dataset**.

### Dataset Details

|     Information       | Value |
|---                    |   ---:|
| Total transactions    | **284,807** |
| Fraudulent transactions | **492** |
| Input features        | **30** |
| Features              | `Time`, `V1`–`V28`, `Amount` |
| Target                | `Class` |

The dataset is highly imbalanced, with fraudulent transactions representing a very small proportion of the total transactions.

> **Note:** The original dataset is not included in this repository.

---

## 🔐 Unsupervised Learning Approach

The most important aspect of this project is that the Autoencoder is trained using **normal transactions only**.

The `Class` fraud label is **not used during Autoencoder training**.

The development process is:

```text
Credit Card Dataset
        ↓
Data Preparation
        ↓
Development / Test Split
        ↓
Select Normal Transactions
        ↓
Normal Training / Validation
        ↓
StandardScaler
        ↓
Train Autoencoder
        ↓
Learn Normal Transaction Patterns

Fraud labels are used only during the final evaluation of the trained models.

🧠 Autoencoder Architecture

The Autoencoder architecture is:

Input: 30 features
       ↓
Dense: 128
       ↓
Dense: 64
       ↓
Dense: 32
       ↓
Bottleneck: 16
       ↓
Dense: 32
       ↓
Dense: 64
       ↓
Dense: 128
       ↓
Output: 30 features

Total parameters: 29,678

The bottleneck layer learns a compact representation of normal transaction patterns.

🚨 Anomaly Detection

After training, each transaction is passed through the Autoencoder.

The original transaction is compared with its reconstructed version.

The difference is measured using reconstruction error.

Transaction
     ↓
Autoencoder
     ↓
Reconstructed Transaction
     ↓
Reconstruction Error
     ↓
Compare with Threshold
     ↓
Normal / Anomaly
Decision Rule

The final anomaly threshold is:

0.21305744

If Reconstruction Error > 0.21305744
        → Anomaly

Otherwise
        → Normal

An anomalous transaction represents a transaction that differs significantly from the normal patterns learned by the Autoencoder and should be considered for further investigation.

⚖️ Threshold Selection

The threshold was selected using reconstruction errors from normal validation transactions.


| Percentile |      Threshold | Normal Transactions Flagged | False Alarm Rate |
| ---------- | -------------: | --------------------------: | ---------------: |
| 95%        |     0.06258296 |                       2,275 |          5.0010% |
| 97.5%      |     0.11462092 |                       1,138 |          2.5016% |
| **99%**    | **0.21305744** |                     **455** |      **1.0002%** |
| 99.5%      |     0.30817743 |                         228 |          0.5012% |


The 99th percentile was selected because it maintained approximately a 1% false-alarm rate on normal validation transactions while providing strong fraud detection performance on the held-out test data.

🧪 Model Evaluation

The final models were evaluated on a held-out test set containing:

56,962 total transactions
56,864 normal transactions
98 known fraud cases

Fraud labels were used only for evaluation.

Final Model Comparison
  Model	        Precision	Recall	 F1-Score	Fraud Detected	Flagged Transactions
Autoencoder	 0.1269	    0.8367	  0.2204	    82/98	            646
One-Class SVM    0.1246	    0.8265	  0.2166	    81/98	            650
Isolation Forest 0.0984	    0.6735	  0.1717	    66/98	            671

🏆 Final Model

The Autoencoder was selected as the final model because it achieved the strongest overall performance among the three evaluated approaches.

Autoencoder Results
Precision: 0.1269
Recall: 0.8367
F1-Score: 0.2204
ROC-AUC: 0.9630
Fraud detected: 82/98
Transactions flagged: 646

The model detected 82 out of 98 known fraud cases in the held-out test set.

⚠️ Limitations

The Autoencoder achieved high recall but relatively low precision.

Out of 646 flagged transactions, 82 were known fraud cases.

Therefore:

646 flagged transactions
− 82 known fraud cases
= 564 false positives

This means some legitimate transactions are also flagged as anomalies.

Therefore, the system is designed as a fraud screening and investigation tool, rather than a system that automatically confirms every flagged transaction as fraud.

🖥️ FraudGuard Streamlit Application

The trained model is integrated into a Streamlit application called FraudGuard.

The application provides three main sections:

1. Fraud Analytics

Provides an overview of the fraud detection results and model performance.

2. Transaction Investigation

Allows existing transactions to be investigated using their anomaly information and reconstruction error.

3. New Transaction Screening

Allows a new transaction to be screened using the trained Autoencoder.

The screening process is:

New Transaction
       ↓
StandardScaler
       ↓
Trained Autoencoder
       ↓
Reconstruction Error
       ↓
Threshold = 0.21305744
       ↓
Normal / Anomaly

When the reconstruction error exceeds the threshold, the transaction is flagged as anomalous and the application provides the reason for the decision.

🛠️ Technologies Used
Python
TensorFlow / Keras
Scikit-learn
Pandas
NumPy
Matplotlib
Streamlit
Joblib
Pickle

📁 Project Structure
Fraud_Detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── autoencoder.keras
│   ├── baseline_thresholds.pkl
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── project_config.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
│
├── notebooks/
│   └── Fraud_Detection_Autoencoder.ipynb
│
├── results/
│   ├── flagged_transactions.csv
│   ├── test_predictions.csv
│   └── threshold_analysis.csv
│
├── src/
│   ├── predictor.py
│   └── test_predictor.py
│
├── docs/
│   ├── Model_Development.docx
│   ├── Evaluation_Report.docx
│   └── Project_Writeup.docx
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
▶️ How to Run
1. Clone the repository
git clone https://github.com/KannetiDivya/AIML_CAPSTONE_PROJECT.git
cd Fraud_Detection
2. Create and activate a virtual environment
python -m venv .venv

Windows:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run FraudGuard
streamlit run app.py

The FraudGuard application will open in your browser.

📂 Project Documentation

Detailed project documentation is available in the docs/ folder:

Step 1 — Model Development: Model architecture, training approach, threshold selection, and final model comparison.
Step 2 — Evaluation Report: Precision, recall, F1-score, fraud detection results, threshold analysis, and model evaluation.
Step 3 — Project Write-up: Complete project methodology, results, limitations, Streamlit application, and conclusion.

📌 Results Summary

The final Autoencoder achieved:

82/98 fraud cases detected

with:

83.67% Recall

12.69% Precision

0.2204 F1-Score

0.9630 ROC-AUC

The results show that the Autoencoder can identify unusual transaction patterns without requiring fraud labels during training.

🔄 Overall Project Flow

Credit Card Dataset
        ↓
Data Preparation
        ↓
Development / Test Split
        ↓
Normal Transactions Only
        ↓
Feature Scaling
        ↓
Train Autoencoder
        ↓
Learn Normal Patterns
        ↓
Calculate Reconstruction Error
        ↓
Select 99th Percentile Threshold
        ↓
Flag Anomalous Transactions
        ↓
Evaluate on Held-Out Test Data
        ↓
Compare with Isolation Forest
and One-Class SVM
        ↓
Select Autoencoder
        ↓
FraudGuard Streamlit Application

🏁 Conclusion

This project demonstrates an unsupervised Autoencoder-based approach for detecting unusual credit card transactions.

The Autoencoder was trained exclusively on normal transactions and used reconstruction error to identify anomalous transactions.

Among the evaluated models, the Autoencoder achieved the best overall performance, detecting 82 out of 98 known fraud cases with a recall of 83.67%.

The final FraudGuard Streamlit application provides a practical interface for transaction screening and investigation.

The system is intended to help identify potentially suspicious transactions for further review rather than automatically confirming transactions as fraudulent.

👩‍💻 Author

Divya Kanneti

AIML_CAPSTONE_PROJECT — Fraud Detection Using Autoencoders
