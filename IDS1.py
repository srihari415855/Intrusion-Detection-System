import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Page Config
st.set_page_config(page_title="IDS", layout="wide")
st.title("Intrusion Detection System")

# 1. DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv('NSL_KDD_Processed_compressed.csv', low_memory=False)
    df = df.dropna()
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna()
    return df

df = load_data()

# Session State Management
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'show_eda' not in st.session_state:
    st.session_state.show_eda = False

# --- SECTION 1: TRAINING (LOGISTIC REGRESSION) ---
st.header("1. Model Training")
if st.button("Train IDS Model"):
    with st.spinner("Training Model..."):
        X = df.drop(columns=['outcome'])
        y_binary = df['outcome'].apply(lambda x: 1 if x != 11.0 else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
        
        model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.success("Training model Complete!")

st.divider()

# --- SECTION 2: COMPLETE EDA REPORT ---
st.header("2. Exploratory Data Analysis (EDA) Report")
if st.button("Show Complete EDA Report"):
    st.session_state.show_eda = True

if st.session_state.show_eda:
    # Normal vs Malicious Calculation
    st.subheader("Traffic Classification Analysis")
    df['traffic_type'] = df['outcome'].apply(lambda x: 'Normal' if x == 11.0 else 'Malicious')
    traffic_counts = df['traffic_type'].value_counts()
    
    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        st.write("**Traffic Summary Table**")
        st.dataframe(traffic_counts)
    with col_e2:
        fig_traffic, ax_traffic = plt.subplots(figsize=(8, 4))
        sns.barplot(x=traffic_counts.index,
                    y=traffic_counts.values,
                    hue=traffic_counts.index,
                    palette=['#2ecc71', '#e74c3c'],
                    legend=False,
                    ax=ax_traffic
            )
        plt.title("Distribution of Normal vs Malicious Traffic")
        st.pyplot(fig_traffic)

    # Variable Profiling
    st.subheader("Variable Analysis")
    var_to_show = st.selectbox("Select Variable to Profile", df.drop(columns=['traffic_type']).columns, key='var_selector')
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        st.write(f"**Stats for {var_to_show}**")
        st.write(df[var_to_show].describe())
    with v_col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[var_to_show], kde=True, bins=30, color='royalblue')
        st.pyplot(fig)

    # Interactions & Correlations
    st.subheader("Interactions & Correlations")
    tab1, tab2 = st.tabs(["Correlation Matrix", "Interactions"])
    
    with tab1:
        fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
        corr = df.select_dtypes(include=[np.number]).iloc[:, :20].corr() 
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        st.pyplot(fig_corr)
        
    with tab2:
        st.write("**Feature Interaction Density**")
        # Selecting two prominent features for interaction
        col_x = st.selectbox("Select X axis", df.columns[:10], index=4) # default src_bytes
        col_y = st.selectbox("Select Y axis", df.columns[:10], index=5) # default dst_bytes
        
        fig_hex, ax_hex = plt.subplots(figsize=(10, 7))
        hb = ax_hex.hexbin(df[col_x], df[col_y], gridsize=30, cmap='Blues', mincnt=1)
        fig_hex.colorbar(hb, ax=ax_hex, label='Frequency Density')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f" Interactions : {col_x} vs {col_y}")
        st.pyplot(fig_hex)

st.divider()

# --- SECTION 3: TESTING & EVALUATION ---
st.header("3. IDS Testing & Evaluation")
if st.session_state.model is not None:
    if st.button("Run Test & Evaluation"):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = model.predict(X_test)
        
        st.metric("Test Accuracy Score", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
            
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=True,
                        xticklabels=['Normal', 'Malicious'],
                        yticklabels=['Normal', 'Malicious'])
            st.pyplot(fig_cm)
            
        with col_res2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=['Normal', 'Malicious'], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())