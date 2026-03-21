import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, roc_curve, auc,
                             precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .improvement-badge {
        background: #00b894;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .section-header {
        border-left: 4px solid #e94560;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Intrusion Detection System")

# ─── Constants ────────────────────────────────────────────────────────────────
NORMAL_LABEL = 11.0

# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('NSL_KDD_Processed_compressed.csv', low_memory=False)
    df = df.dropna()
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna()
    df['traffic_type'] = df['outcome'].apply(
        lambda x: 'Normal' if x == NORMAL_LABEL else 'Malicious'
    )
    return df

@st.cache_data
def engineer_features(df):
    """
    IMPROVEMENT 2: Feature Engineering
    Instead of compressing with PCA (which loses info),
    we create meaningful domain-specific ratios from raw features.
    """
    df = df.copy()

    # Byte-level ratios — reveal exfiltration patterns
    if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
        df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        df['total_bytes'] = df['src_bytes'] + df['dst_bytes']

    # Connection rate features — detect scanning/DoS
    if 'duration' in df.columns:
        if 'src_bytes' in df.columns:
            df['bytes_per_second'] = df['src_bytes'] / (df['duration'] + 1)
        if 'count' in df.columns:
            df['connections_per_second'] = df['count'] / (df['duration'] + 1)

    # Login failure ratio — brute-force indicator
    if 'num_failed_logins' in df.columns and 'count' in df.columns:
        df['failed_login_ratio'] = df['num_failed_logins'] / (df['count'] + 1)

    # Error rate combination
    if 'serror_rate' in df.columns and 'rerror_rate' in df.columns:
        df['combined_error_rate'] = df['serror_rate'] + df['rerror_rate']

    # Same-host / same-service ratio (useful for port scan detection)
    if 'same_srv_rate' in df.columns and 'diff_srv_rate' in df.columns:
        df['srv_diversity'] = df['diff_srv_rate'] / (df['same_srv_rate'] + 0.01)

    return df

df_raw = load_data()
df = engineer_features(df_raw)

# ─── Session State ────────────────────────────────────────────────────────────
for key, default in [
    ('model', None), ('X_test', None), ('y_test', None),
    ('y_pred', None), ('y_prob', None), ('show_eda', False),
    ('scaler', None), ('best_threshold', 0.5),
    ('feature_names', None), ('train_results', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── SECTION 1: EDA ───────────────────────────────────────────────────────────
st.header("1. Exploratory Data Analysis")
if st.button("Show Complete EDA Report"):
    st.session_state.show_eda = True

if st.session_state.show_eda:
    st.subheader("Traffic Classification Analysis")
    traffic_counts = df['traffic_type'].value_counts()
    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        traffic_percent = df['traffic_type'].value_counts(normalize=True) * 100
        traffic_summary = pd.DataFrame({
            "Count": traffic_counts,
            "Percentage (%)": traffic_percent.round(2)
        })
        st.dataframe(traffic_summary)

        # Imbalance warning
        ratio = traffic_counts.min() / traffic_counts.max()
        if ratio < 0.4:
            st.warning(f"⚠️ Class imbalance detected (ratio={ratio:.2f}). SMOTE will be applied during training.")
        else:
            st.success(f"✅ Class balance is acceptable (ratio={ratio:.2f})")

    with col_e2:
        fig_traffic, ax_traffic = plt.subplots(figsize=(8, 4))
        sns.barplot(x=traffic_counts.index, y=traffic_counts.values,
                    hue=traffic_counts.index,
                    palette=['#2ecc71', '#e74c3c'], legend=False, ax=ax_traffic)
        ax_traffic.set_title("Distribution of Normal vs Malicious Traffic")
        st.pyplot(fig_traffic)
        plt.close(fig_traffic)

    st.subheader("Engineered Features Preview")
    engineered_cols = [c for c in df.columns if c not in df_raw.columns]
    if engineered_cols:
        st.dataframe(df[engineered_cols + ['traffic_type']].head(10))
    else:
        st.info("No engineered features found — check column names match NSL-KDD format.")

    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing Percentage (%)": missing_percent.round(2)
    })
    if missing_df["Missing Count"].sum() == 0:
        st.success("✅ No Missing Values Found in the Data")
    else:
        st.dataframe(missing_df[missing_df["Missing Count"] > 0])

    st.subheader("Variable Analysis")
    var_to_show = st.selectbox("Select Variable to Profile",
                                df.drop(columns=['traffic_type']).columns,
                                key='var_selector')
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        st.write(f"**Stats for `{var_to_show}`**")
        st.write(df[var_to_show].describe())
    with v_col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[var_to_show], kde=True, bins=30, color='royalblue', ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Matrix (top 20 features)")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    corr = df.select_dtypes(include=[np.number]).iloc[:, :20].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax_corr)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

st.divider()

# ─── SECTION 2: MODEL TRAINING ────────────────────────────────────────────────
st.header("2. Model Training")

# XGBoost hyperparameter controls
st.subheader("XGBoost Hyperparameters")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    n_estimators = st.slider("Number of trees", 100, 500, 300, 50)
with col_p2:
    max_depth = st.slider("Max depth", 3, 10, 6)
with col_p3:
    learning_rate = st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
with col_p4:
    use_smote = st.checkbox("Apply SMOTE", value=True)

col_p5, col_p6 = st.columns(2)
with col_p5:
    subsample = st.slider("Subsample ratio", 0.5, 1.0, 0.8, 0.05)
with col_p6:
    colsample = st.slider("Column sample ratio", 0.5, 1.0, 0.8, 0.05)

if st.button("Train IDS Model", type="primary"):
    with st.spinner("Training IDS model..."):

        # 1. Prepare features (drop non-feature cols)
        drop_cols = ['outcome', 'traffic_type']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        y_binary = df['outcome'].apply(lambda x: 1 if x != NORMAL_LABEL else 0)

        feature_names = X.columns.tolist()
        st.session_state.feature_names = feature_names

        # 2. Train/test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )

        # 3. Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        st.session_state.scaler = scaler

        # 4. SMOTE (IMPROVEMENT 3)
        if use_smote:
            progress_text = st.empty()
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_train_final, y_train_final = sm.fit_resample(X_train_scaled, y_train)
            progress_text.success(
                f"✅ SMOTE applied: {y_train.sum()} → {y_train_final.sum()} malicious samples"
            )
        else:
            X_train_final, y_train_final = X_train_scaled, y_train

        X_test_final = X_test_scaled

        # 5. Class weight for XGBoost (backup balancing even without SMOTE)
        neg_count = (y_train_final == 0).sum()
        pos_count = (y_train_final == 1).sum()
        scale_pos_weight = neg_count / pos_count

        # 6. XGBoost model (IMPROVEMENT 1)
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_test_final, y_test)],
            verbose=False
        )

        train_accuracy = model.score(X_train_final, y_train_final)
        st.session_state.model = model

        # 7. Cross-validation score
        cv_scores = cross_val_score(model, X_train_final, y_train_final,
                                     cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)

        # 8. Threshold tuning (IMPROVEMENT 4)
        y_prob_train = model.predict_proba(X_test_final)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = [f1_score(y_test, (y_prob_train >= t).astype(int), zero_division=0)
                     for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        st.session_state.best_threshold = best_threshold

        # Save test data
        st.session_state.X_test = X_test_final
        st.session_state.y_test = y_test

        # Store training results for display
        st.session_state.train_results = {
            'train_accuracy': train_accuracy,
            'cv_scores': cv_scores,
            'best_threshold': best_threshold,
            'max_f1': max(f1_scores),
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'X_train_final': X_train_final,
            'y_train_final': y_train_final,
            'evals_result': model.evals_result() if hasattr(model, 'evals_result') else None
        }

        st.success("XGBoost model trained successfully!")

# Display training results if available
if st.session_state.train_results is not None:
    r = st.session_state.train_results
    cv_scores = r['cv_scores']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training Accuracy", f"{r['train_accuracy']*100:.2f}%")
    m2.metric("CV F1 Score (mean)", f"{cv_scores.mean():.4f}", f"±{cv_scores.std():.4f}")
    m3.metric("Optimal Threshold", f"{r['best_threshold']:.2f}",
              f"Default was 0.50" if r['best_threshold'] != 0.5 else "No change")
    m4.metric("Best F1 @ threshold", f"{r['max_f1']:.4f}")

    # Threshold tuning plot
    st.subheader("Threshold Tuning — F1 vs Decision Threshold")
    fig_thresh, ax_thresh = plt.subplots(figsize=(10, 4))
    ax_thresh.plot(r['thresholds'], r['f1_scores'], color='#3498db', linewidth=2)
    ax_thresh.axvline(x=r['best_threshold'], color='#e74c3c', linestyle='--',
                      label=f"Best threshold = {r['best_threshold']:.2f}")
    ax_thresh.axvline(x=0.5, color='#95a5a6', linestyle=':', label='Default threshold = 0.50')
    ax_thresh.set_xlabel("Decision Threshold")
    ax_thresh.set_ylabel("F1 Score")
    ax_thresh.set_title("F1 Score vs Classification Threshold")
    ax_thresh.legend()
    ax_thresh.grid(True, alpha=0.3)
    st.pyplot(fig_thresh)
    plt.close(fig_thresh)

    # Learning curve (XGBoost eval log)
    model = st.session_state.model
    if hasattr(model, 'evals_result_') and model.evals_result_:
        evals = model.evals_result_
        if 'validation_0' in evals and 'logloss' in evals['validation_0']:
            st.subheader("Learning Curve (Validation Log Loss)")
            fig_lc, ax_lc = plt.subplots(figsize=(10, 4))
            ax_lc.plot(evals['validation_0']['logloss'], color='#e74c3c',
                       linewidth=1.5, label='Validation loss')
            ax_lc.set_xlabel("Boosting Round")
            ax_lc.set_ylabel("Log Loss")
            ax_lc.set_title("XGBoost Learning Curve")
            ax_lc.legend()
            ax_lc.grid(True, alpha=0.3)
            st.pyplot(fig_lc)
            plt.close(fig_lc)

st.divider()

# ─── SECTION 3: TESTING & EVALUATION ─────────────────────────────────────────
st.header("3. IDS Testing & Evaluation")

if st.session_state.model is not None:
    if st.button("▶ Run Test & Evaluation", type="primary"):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        best_threshold = st.session_state.best_threshold

        # Predictions using tuned threshold
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_tuned = (y_prob >= best_threshold).astype(int)
        y_pred_default = (y_prob >= 0.5).astype(int)

        st.session_state.y_pred = y_pred_tuned
        st.session_state.y_prob = y_prob

        # Comparison metrics
        st.subheader("Default (0.5) vs Tuned Threshold Comparison")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            acc_default = accuracy_score(y_test, y_pred_default)
            acc_tuned = accuracy_score(y_test, y_pred_tuned)
            delta = acc_tuned - acc_default
            st.metric("Accuracy (tuned)", f"{acc_tuned*100:.2f}%",
                      f"{delta*100:+.2f}% vs default")
        with comp_col2:
            f1_default = f1_score(y_test, y_pred_default)
            f1_tuned = f1_score(y_test, y_pred_tuned)
            st.metric("F1 Score (tuned)", f"{f1_tuned:.4f}",
                      f"{f1_tuned - f1_default:+.4f} vs default")
        with comp_col3:
            st.metric("Threshold used", f"{best_threshold:.2f}",
                      "Optimized for F1")

        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_tuned)
            tn, fp, fn, tp = cm.ravel()
            labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=True,
                        xticklabels=['Normal', 'Malicious'],
                        yticklabels=['Normal', 'Malicious'],
                        ax=ax_cm)
            ax_cm.set_title(f"Confusion Matrix (threshold={best_threshold:.2f})")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_res2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred_tuned,
                                           target_names=['Normal', 'Malicious'],
                                           output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4))

        # ROC + Precision-Recall curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresh = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        curve_col1, curve_col2 = st.columns(2)
        with curve_col1:
            st.subheader(f"ROC Curve (AUC = {roc_auc:.4f})")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, label=f"XGBoost AUC = {roc_auc:.4f}", color='#3498db', lw=2)
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        with curve_col2:
            st.subheader(f"Precision-Recall Curve (AUC = {pr_auc:.4f})")
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            ax_pr.plot(recall, precision, color='#e74c3c', lw=2,
                       label=f"PR AUC = {pr_auc:.4f}")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall Curve")
            ax_pr.legend()
            ax_pr.grid(True, alpha=0.3)
            st.pyplot(fig_pr)
            plt.close(fig_pr)

        # Feature Importance (XGBoost native — much more meaningful than PCA coefs)
        st.subheader("Feature Importance (XGBoost — Gain)")
        feature_names = st.session_state.feature_names
        if feature_names:
            importance_gain = model.get_booster().get_score(importance_type='gain')
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance (Gain)": v}
                for k, v in importance_gain.items()
            ]).sort_values("Importance (Gain)", ascending=False).head(20)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            colors = ['#e74c3c' if '_ratio' in f or 'per_' in f or 'combined' in f
                      else '#3498db' for f in imp_df["Feature"]]
            ax_imp.barh(imp_df["Feature"], imp_imp := imp_df["Importance (Gain)"], color=colors)
            ax_imp.invert_yaxis()
            ax_imp.set_title("Top 20 Most Important Features")
            ax_imp.set_xlabel("Gain")
            legend_patches = [
                mpatches.Patch(color='#e74c3c', label='Engineered feature'),
                mpatches.Patch(color='#3498db', label='Original feature')
            ]
            ax_imp.legend(handles=legend_patches)
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            engineered_in_top20 = [f for f in imp_df["Feature"]
                                   if any(x in f for x in ['_ratio','per_','combined','total_','diversity'])]
            if engineered_in_top20:
                st.success(f"Engineered features in top 20: `{'`, `'.join(engineered_in_top20)}`")

        # Probability distribution
        st.subheader("Prediction Confidence Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
        normal_probs = y_prob[y_test == 0]
        malicious_probs = y_prob[y_test == 1]
        ax_dist.hist(normal_probs, bins=50, alpha=0.6, color='#2ecc71',
                     label='Normal traffic', density=True)
        ax_dist.hist(malicious_probs, bins=50, alpha=0.6, color='#e74c3c',
                     label='Malicious traffic', density=True)
        ax_dist.axvline(x=best_threshold, color='black', linestyle='--',
                        label=f'Optimal threshold = {best_threshold:.2f}')
        ax_dist.set_xlabel("Predicted Probability (Malicious)")
        ax_dist.set_ylabel("Density")
        ax_dist.set_title("Confidence Score Separation — Normal vs Malicious")
        ax_dist.legend()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

else:
    st.info("⬆️ Please train the model first")

st.divider()

