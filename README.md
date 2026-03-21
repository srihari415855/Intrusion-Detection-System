# 🛡️ Intrusion Detection System (IDS) v3 — XGBoost Enhanced

An interactive, ML-powered **Network Intrusion Detection System** built with Streamlit. It classifies network traffic as **Normal** or **Malicious** using XGBoost with advanced feature engineering, SMOTE-based class balancing, and F1-optimal threshold tuning — trained on the **NSL-KDD** benchmark dataset.

---

## 📌 Project Description

This IDS is a full-stack machine learning application that detects malicious network activity in real time. It is built as an interactive web dashboard where users can explore the dataset, train an XGBoost model with tunable hyperparameters, evaluate its performance with rich visualizations, and simulate live traffic classification.

The project evolved from a v2 baseline (Logistic Regression + PCA) to **v3**, introducing four key improvements:

| Area | v2 (Old) | v3 (This Project) |
|---|---|---|
| **Algorithm** | Logistic Regression | XGBoost (gradient-boosted trees) |
| **Features** | PCA compression | Raw features + engineered domain ratios |
| **Class Imbalance** | `class_weight` only | SMOTE oversampling |
| **Decision Threshold** | Fixed at 0.5 | F1-optimal threshold tuning |

---

## 🚀 Features

- **Exploratory Data Analysis (EDA)**
  - Traffic distribution (Normal vs. Malicious)
  - Class imbalance detection with automatic SMOTE warning
  - Engineered features preview
  - Missing value analysis
  - Per-variable statistical profiling and histograms
  - Correlation heatmap (top 20 features)

- **Model Training**
  - XGBoost classifier with fully adjustable hyperparameters (number of trees, max depth, learning rate, subsample ratios)
  - SMOTE toggle to synthetically oversample the minority class
  - 5-fold Stratified Cross-Validation (F1 scoring)
  - Automatic F1-optimal decision threshold tuning
  - XGBoost learning curve (validation log loss)

- **Testing & Evaluation**
  - Confusion matrix (TN / FP / FN / TP)
  - Classification report (Precision, Recall, F1)
  - ROC Curve with AUC
  - Precision-Recall Curve with AUC
  - Feature Importance (XGBoost Gain) — highlights engineered features
  - Prediction confidence distribution (Normal vs. Malicious separation)
  - Comparison of tuned vs. default (0.5) threshold

- **Live Traffic Prediction**
  - Input real feature values or select from presets (Normal, Port Scan, DoS)
  - Instant classification with confidence score
  - Interactive Plotly gauge chart showing malicious probability

---

## 🧠 Feature Engineering

Beyond raw NSL-KDD features, the following domain-specific ratios are computed to improve detection:

| Feature | Purpose |
|---|---|
| `byte_ratio` | `src_bytes / (dst_bytes + 1)` — detects data exfiltration |
| `total_bytes` | `src_bytes + dst_bytes` — overall traffic volume |
| `bytes_per_second` | `src_bytes / (duration + 1)` — connection throughput |
| `connections_per_second` | `count / (duration + 1)` — scanning / DoS detection |
| `failed_login_ratio` | `num_failed_logins / (count + 1)` — brute-force indicator |
| `combined_error_rate` | `serror_rate + rerror_rate` — overall error signal |
| `srv_diversity` | `diff_srv_rate / (same_srv_rate + 0.01)` — port-scan indicator |

---

## 📂 Project Structure

```
├── Intrusion-Detection-System.py   # Main Streamlit application
├── NSL_KDD_Processed_compressed.csv  # Dataset (required, see below)
└── README.md
```

---

## 📦 Requirements

**Python 3.8+**

Install all dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly xgboost imbalanced-learn
```

Or use a `requirements.txt`:

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
xgboost
imbalanced-learn
```

---

## 🗂️ Dataset

This project uses the **NSL-KDD** dataset — an improved version of the original KDD Cup 1999 dataset, widely used for IDS benchmarking.

- Download from: [NSL-KDD on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) or the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)
- The app expects a pre-processed CSV file named `NSL_KDD_Processed_compressed.csv` in the same directory as the script
- The `outcome` column encodes traffic labels; `11.0` is treated as **Normal**, all other values as **Malicious**

---

## ▶️ Running the App

```bash
py -m streamlit run Intrusion-Detection-System.py
```

Then open your browser at `http://localhost:8501`.

---

## 🖥️ App Workflow

1. **EDA** — Click *Show Complete EDA Report* to explore the dataset
2. **Train** — Adjust XGBoost hyperparameters, enable/disable SMOTE, then click *Train XGBoost IDS Model*
3. **Evaluate** — Click *Run Test & Evaluation* to see metrics, curves, and feature importances
4. **Predict** — Use the *Live Traffic Prediction* section to classify individual connections

---

## 📊 Model Performance (Typical Results on NSL-KDD)

| Metric | Value |
|---|---|
| Accuracy | ~99% |
| F1 Score | ~0.99 |
| ROC-AUC | ~0.999 |
| CV F1 (5-fold) | ~0.99 ± 0.001 |

> Results may vary based on hyperparameter choices and SMOTE settings.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `Streamlit` | Interactive web dashboard |
| `XGBoost` | Gradient-boosted tree classifier |
| `scikit-learn` | Preprocessing, metrics, cross-validation |
| `imbalanced-learn` | SMOTE oversampling |
| `Plotly` | Gauge chart for live predictions |
| `Matplotlib / Seaborn` | Static visualizations |
| `Pandas / NumPy` | Data manipulation |

---

## 📈 Key Design Decisions

- **XGBoost over Logistic Regression**: Captures non-linear relationships in network traffic patterns without manual feature transformation.
- **Feature Engineering over PCA**: Domain-specific ratios preserve interpretability and improve detection of specific attack types (DoS, port scans, brute force).
- **SMOTE over class_weight**: Generates synthetic minority-class samples in feature space rather than simply re-weighting, leading to better recall on rare attack types.
- **Threshold Tuning**: The decision boundary is shifted from the default 0.5 to the value that maximizes the F1 score on the test set, balancing precision and recall for security contexts where both false positives and false negatives are costly.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙋 Author

Built as a machine learning security project. Contributions and feedback are welcome — feel free to open an issue or pull request.
