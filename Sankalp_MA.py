# -*- coding: utf-8 -*-
"""loan_default.py - Final Code with Confusion Matrix and Theme 1 (Bright)"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score 
import numpy as np

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Loan Default Marketing Analytics", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    try:
        # NOTE: Ensure 'loan_default.xlsx' is in the same folder as this script
        df = pd.read_excel('Loan_Sankalp.xlsx')
        return df
    except FileNotFoundError:
        st.error("File 'loan_default.xlsx' not found. Please place the excel in the directory.")
        return None

df = load_data()

if df is not None:
    # --- 2. DATA PREPROCESSING & MODELING (Risk Scoring) ---
    
    df_model = df.copy()
    le_dict = {}
    for col in df_model.columns:
        le = LabelEncoder()
        # Handle potential NaNs in categorical columns by converting to string
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    # Define X (Features) and y (Target)
    X = df_model.drop('Default', axis=1)
    y = df_model['Default']

    # Train a simple Decision Tree (max_depth=5 to prevent overfitting)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    # Get probabilities for the 'Yes' class (Risk Score)
    yes_index = list(le_dict['Default'].classes_).index('Yes')
    df['Risk_Probability'] = clf.predict_proba(X)[:, yes_index]

    # --- NEW: Calculate Confusion Matrix and Metrics ---
    y_pred = clf.predict(X)
    
    # Calculate the Confusion Matrix (format: [[TN, FP], [FN, TP]])
    cm = confusion_matrix(y, y_pred)
    
    # Extract values
    TN = cm[0, 0] # True Negative
    FP = cm[0, 1] # False Positive
    FN = cm[1, 0] # False Negative
    TP = cm[1, 1] # True Positive
    
    # Calculate Metrics
    accuracy = accuracy_score(y, y_pred)
    
    # Check for division by zero before calculating rates
    total_actual_no = TN + FP
    total_actual_yes = FN + TP

    type_i_error_rate = FP / total_actual_no if total_actual_no > 0 else 0
    type_ii_error_rate = FN / total_actual_yes if total_actual_yes > 0 else 0


    # --- 3. DASHBOARD LAYOUT ---

    st.title("Loan Default Marketing Analytics Dashboard")
    st.markdown("Analysis of customer profiles and model performance to identify high-risk segments.")
    st.divider()

    # --- TOP ROW: KPI (Updated with Model Metrics) ---
    total_defaults = df[df['Default'] == 'Yes'].shape[0]
    total_customers = df.shape[0]
    default_rate = (total_defaults / total_customers) * 100

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(label="Overall Default Rate", value=f"{default_rate:.1f}%", delta_color="inverse")
    with col_kpi2:
        st.metric(label="Model Accuracy", value=f"{accuracy*100:.1f}%")
    with col_kpi3:
        st.metric(label="False Pos. Rate (Type I Error)", value=f"{type_i_error_rate*100:.1f}%")
    with col_kpi4:
        st.metric(label="False Neg. Rate (Type II Error)", value=f"{type_ii_error_rate*100:.1f}%")

    st.divider()

    # --- MIDDLE ROW: THE DRIVERS (Bar Charts - Theme 1 Applied) ---
    st.subheader("The Drivers: Risk Analysis")

    row2_col1, row2_col2 = st.columns(2)

    # Helper function to calculate risk rate by category
    def get_risk_by_category(column_name):
        risk_df = df.groupby(column_name)['Default'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        risk_df.columns = [column_name, 'Default Rate (%)']
        return risk_df.sort_values('Default Rate (%)', ascending=False)

    # Chart 1: Risk by Employment
    with row2_col1:
        emp_risk = get_risk_by_category('Employment_Type')
        fig_emp = px.bar(emp_risk, x='Employment_Type', y='Default Rate (%)',
                         title="Risk by Employment Type",
                         color='Default Rate (%)', 
                         color_continuous_scale='Plasma') # <--- THEME 1: PLASMA
        st.plotly_chart(fig_emp, use_container_width=True)

    # Chart 2: Risk by Credit History
    with row2_col2:
        cred_risk = get_risk_by_category('Credit_History')
        fig_cred = px.bar(cred_risk, x='Credit_History', y='Default Rate (%)',
                          title="Risk by Credit History",
                          color='Default Rate (%)', 
                          color_continuous_scale='Plasma') # <--- THEME 1: PLASMA
        st.plotly_chart(fig_cred, use_container_width=True)

    st.divider()
    
    # --- CONFUSION MATRIX ANALYSIS ---
    st.subheader("Model Performance: Confusion Matrix")
    
    cm_col, explanation_col = st.columns([1, 2])

    with cm_col:
        st.markdown("**Predicted vs. Actual Outcomes**")
        # Create a DataFrame for displaying the matrix beautifully
        cm_df = pd.DataFrame(cm, 
                             index=['Actual No Default (No)', 'Actual Default (Yes)'], 
                             columns=['Predicted No Default (No)', 'Predicted Default (Yes)'])
        st.dataframe(cm_df)
    
    with explanation_col:
        st.markdown("**Understanding the Confusion Matrix Values**")
        st.markdown(f"""
        This matrix evaluates the **Decision Tree Model's** performance in classifying customers:
        
        * **True Negative (TN): {TN}** - The model **Correctly** predicted **No Default**. (Good)
        * **True Positive (TP): {TP}** - The model **Correctly** predicted **Default (Yes)**. (Good)
        * **False Positive (FP): {FP}** - The model **Incorrectly** predicted **Default (Yes)** when the customer *didn't* default. This is a **Type I Error** (Risk: Turning away a good customer).
        * **False Negative (FN): {FN}** - The model **Incorrectly** predicted **No Default** when the customer *did* default. This is a **Type II Error** (Major Risk: Approving a loan that will default).
        
        A good model for loan risk minimizes **False Negatives (FN)**.
        """)
        
    st.divider()

    # --- BOTTOM ROW: THE PROFILE (Table & Insight) ---
    st.subheader("The Profile: High Risk Identification")

    row3_col1, row3_col2 = st.columns([2, 1])

    with row3_col1:
        st.markdown("**Top 5 Riskiest Profiles (Highest Predicted Probability)**")
        # Sort by Risk Probability (Descending) and take top 5
        top_risky = df.sort_values(by='Risk_Probability', ascending=False).head(5)

        # Select columns to display suitable for a report
        display_cols = ['Employment_Type', 'Credit_History', 'Income_Bracket', 'Education_Level', 'Risk_Probability', 'Default']

        # Format the probability as percentage for display
        display_df = top_risky[display_cols].copy()
        display_df['Risk_Probability'] = display_df['Risk_Probability'].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(display_df, use_container_width=True)

    with row3_col2:
        st.info("💡 **AI Insight**")

        # Dynamic Insight Generation
        highest_risk_emp = emp_risk.iloc[0]['Employment_Type']
        highest_risk_cred = cred_risk.iloc[0]['Credit_History']

        insight_text = f"""
        The analysis indicates a strong correlation between employment stability and loan repayment.

        **Key Observations:**
        - **Primary Risk Factor:** Customers with **{highest_risk_emp}** status show the highest propensity to default.
        - **Secondary Signal:** A **{highest_risk_cred}** in credit history is a critical warning sign.

        **Model Check:**
        The model made **{FN} False Negatives**, meaning {FN} customers who *actually defaulted* were incorrectly predicted as safe. This is a primary area for model improvement.
        
        **Recommendation:**
        Marketing campaigns for loans should filter out unemployed profiles with credit delays to lower the overall **{default_rate:.1f}%** default rate.
        """
        st.markdown(insight_text)

else:
    st.warning("Please ensure 'loan_default.xlsx' is in the same directory and run the application using: `streamlit run loan_default.py`")

# To run the app: streamlit run loan_default.py
