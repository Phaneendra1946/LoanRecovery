import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title='Loan Recovery Prediction', layout='wide')

base_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(base_dir, 'loan_recovery_model.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
    le = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
    feature_names = joblib.load(os.path.join(base_dir, 'feature_names.pkl'))
    return model, scaler, le, feature_names

def main():
    st.title('🏦 AI-Based Loan Recovery Probability Prediction')
    st.markdown('Build classification models to predict the likelihood of loan recovery, '
                'evaluate recall and ROC metrics, and support optimized debt recovery '
                'strategies with explainable insights.')

    try:
        model, scaler, le, feature_names = load_assets()
    except Exception as e:
        st.error("⚠️ Model not found. Please run `python generate_dataset.py` then `python train_model.py` first.")
        return

    # ── Sidebar: Borrower Inputs ──
    st.sidebar.header('📋 Borrower Features')
    borrower_age = st.sidebar.number_input('Borrower Age', min_value=21, max_value=65, value=35)
    annual_income = st.sidebar.number_input('Annual Income ($)', min_value=15000.0, max_value=200000.0, value=50000.0, step=1000.0)
    loan_amount = st.sidebar.number_input('Loan Amount ($)', min_value=5000.0, max_value=100000.0, value=20000.0, step=500.0)
    interest_rate = st.sidebar.number_input('Interest Rate (%)', min_value=5.0, max_value=25.0, value=12.0, step=0.5)
    loan_tenure_months = st.sidebar.number_input('Loan Tenure (Months)', min_value=6, max_value=60, value=24)
    credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
    num_previous_defaults = st.sidebar.number_input('Previous Defaults', min_value=0, max_value=5, value=0)
    employment_type = st.sidebar.selectbox('Employment Type', ['Salaried', 'Self-Employed', 'Unemployed', 'Retired'])
    months_since_last_payment = st.sidebar.number_input('Months Since Last Payment', min_value=0, max_value=24, value=3)
    total_paid_amount = st.sidebar.number_input('Total Paid Amount ($)', min_value=0.0, max_value=100000.0, value=5000.0, step=500.0)
    debt_to_income_ratio = st.sidebar.number_input('Debt to Income Ratio', min_value=0.1, max_value=0.8, value=0.3, step=0.05)
    collection_attempts = st.sidebar.number_input('Collection Attempts', min_value=0, max_value=10, value=1)

    if st.sidebar.button('🔍 Predict Recovery Probability'):
        input_data = pd.DataFrame([{
            'borrower_age': borrower_age,
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'loan_tenure_months': loan_tenure_months,
            'credit_score': credit_score,
            'num_previous_defaults': num_previous_defaults,
            'employment_type': employment_type,
            'months_since_last_payment': months_since_last_payment,
            'total_paid_amount': total_paid_amount,
            'debt_to_income_ratio': debt_to_income_ratio,
            'collection_attempts': collection_attempts
        }])

        # Encode employment type
        try:
            input_data['employment_type'] = le.transform(input_data['employment_type'])
        except ValueError:
            input_data['employment_type'] = 0

        # Reorder columns to match training
        input_data = input_data[feature_names]

        # Scale
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_names)

        # Predict
        prob = model.predict_proba(input_scaled)[0, 1]
        pred_class = 'Recovered' if prob >= 0.5 else 'Not Recovered'

        st.subheader('📊 Prediction Result')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", pred_class)
        with col2:
            st.metric("Recovery Probability", f"{prob * 100:.2f}%")
        with col3:
            st.metric("Confidence", f"{max(prob, 1-prob) * 100:.1f}%")

        # Progress bar gauge
        st.progress(float(prob))
        if prob >= 0.7:
            st.success("✅ High recovery probability – gentle reminder may suffice.")
        elif prob >= 0.4:
            st.warning("⚠️ Moderate recovery probability – consider structured follow-up.")
        else:
            st.error("❌ Low recovery probability – debt restructuring may be needed.")

        # SHAP waterfall for individual prediction
        st.subheader('🔬 Individual Prediction Explanation (SHAP)')
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_scaled, check_additivity=False)
            # Handle different shap_values return formats
            if isinstance(sv, list):
                # List of arrays: [class0_array, class1_array]
                vals = sv[1][0]  # class 1, first sample
                base = explainer.expected_value[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                # 3D array: (n_samples, n_features, n_classes)
                vals = sv[0, :, 1]  # first sample, all features, class 1
                base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                # 2D array: (n_samples, n_features)
                vals = sv[0]
                base = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

            exp_obj = shap.Explanation(
                values=vals,
                base_values=float(base),
                data=input_scaled.values[0],
                feature_names=feature_names
            )
            fig, ax = plt.subplots()
            shap.plots.waterfall(exp_obj, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP waterfall could not be rendered: {e}")

    # ── Model Performance & EDA Section ──
    st.markdown("---")
    st.header('📈 Model Performance & Exploratory Data Analysis')
    plots_dir = os.path.join(base_dir, 'plots')

    if os.path.isdir(plots_dir):
        tab1, tab2, tab3 = st.tabs(["ROC & SHAP", "Confusion Matrices", "EDA Plots"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                roc_path = os.path.join(plots_dir, 'roc_curves.png')
                if os.path.exists(roc_path):
                    st.image(roc_path, caption='ROC Curves – All Models')
            with c2:
                shap_path = os.path.join(plots_dir, 'shap_summary.png')
                if os.path.exists(shap_path):
                    st.image(shap_path, caption='SHAP Feature Importance')

        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                p = os.path.join(plots_dir, 'logistic_regression_cm.png')
                if os.path.exists(p):
                    st.image(p, caption='Logistic Regression')
            with c2:
                p = os.path.join(plots_dir, 'random_forest_cm.png')
                if os.path.exists(p):
                    st.image(p, caption='Random Forest')
            with c3:
                p = os.path.join(plots_dir, 'xgboost_cm.png')
                if os.path.exists(p):
                    st.image(p, caption='XGBoost')

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                p = os.path.join(plots_dir, 'recovery_distribution.png')
                if os.path.exists(p):
                    st.image(p, caption='Recovery Status Distribution')
            with c2:
                p = os.path.join(plots_dir, 'correlation_heatmap.png')
                if os.path.exists(p):
                    st.image(p, caption='Feature Correlation Heatmap')
            p = os.path.join(plots_dir, 'repayment_behavior.png')
            if os.path.exists(p):
                st.image(p, caption='Repayment Behavior Analysis')
    else:
        st.info("No plots found. Run train_model.py to generate visualizations.")

    # ── Step 16: Test model using recovery/no-recovery cases ──
    st.markdown("---")
    st.header('🧪 Sample Test Cases (Step 16)')
    st.markdown("Use these pre-filled examples to validate the model:")

    tc1, tc2 = st.columns(2)
    with tc1:
        st.subheader("✅ High Recovery Case")
        st.table(pd.DataFrame([{
            'Borrower Age': 40, 'Annual Income': 120000, 'Loan Amount': 25000,
            'Interest Rate': 8.0, 'Loan Tenure': 24, 'Credit Score': 780,
            'Previous Defaults': 0, 'Employment': 'Salaried',
            'Months Since Payment': 1, 'Total Paid': 18000,
            'Debt/Income Ratio': 0.15, 'Collection Attempts': 1
        }]))
    with tc2:
        st.subheader("❌ Low Recovery Case")
        st.table(pd.DataFrame([{
            'Borrower Age': 28, 'Annual Income': 22000, 'Loan Amount': 45000,
            'Interest Rate': 22.0, 'Loan Tenure': 48, 'Credit Score': 380,
            'Previous Defaults': 4, 'Employment': 'Unemployed',
            'Months Since Payment': 18, 'Total Paid': 3000,
            'Debt/Income Ratio': 0.75, 'Collection Attempts': 9
        }]))

    # ── Step 18: Document business and ethical considerations ──
    st.markdown("---")
    st.header('📝 Business & Ethical Considerations (Step 18)')
    st.markdown("""
    **Business Value:**
    - **Optimized Collection Strategy:** Recovery probability scores help prioritize collection efforts.
      High-probability cases need gentle reminders; low-probability cases may need debt restructuring.
    - **Resource Allocation:** Focus agent time and resources on cases with the highest recovery potential.
    - **Risk Assessment:** Integrate predictions into loan origination to assess future recovery risk.

    **Ethical Considerations:**
    - **Bias & Fairness:** The model must not discriminate based on protected attributes (age, gender,
      ethnicity). Regular fairness audits across demographic segments are essential.
    - **Transparency:** SHAP and LIME explanations ensure predictions can be justified to borrowers,
      regulators, and internal auditors.
    - **Data Privacy:** Borrower data must be handled in compliance with data protection regulations
      (e.g., GDPR, CCPA). Only necessary features should be collected and stored securely.
    - **Human Oversight:** Model predictions should support – not replace – human decision-making.
      Final collection actions should involve human review, especially for edge cases.
    - **Responsible Collection:** Predictions should not be used to harass borrowers. Ethical
      collection practices must be maintained regardless of recovery probability.
    """)

if __name__ == '__main__':
    main()

