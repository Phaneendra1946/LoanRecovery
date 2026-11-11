import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import shap
import lime
import lime.lime_tabular

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── Step 1: Define loan recovery prediction goal ──
    print("=" * 60)
    print("Step 1: GOAL - Predict the likelihood of loan recovery using")
    print("        classification models and support optimized debt")
    print("        recovery strategies with explainable insights.")
    print("=" * 60)

    # ── Step 2: Understand loan lifecycle and recovery process ──
    print("\nStep 2: LOAN LIFECYCLE")
    print("  Origination -> Disbursement -> Repayment -> Default -> Collections -> Recovery/Write-off")
    print("  This model focuses on the Collections -> Recovery stage.\n")

    # ── Step 3: Collect borrower and loan history data ──
    data_path = os.path.join(base_dir, 'loan_recovery_data.csv')
    if not os.path.exists(data_path):
        print("Dataset not found. Run generate_dataset.py first.")
        return
    df = pd.read_csv(data_path)
    print(f"Step 3: Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"  Columns: {list(df.columns)}\n")

    # ── Step 4: Clean missing and inconsistent entries ──
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['loan_id'], inplace=True)
    print(f"Step 4: Cleaned data. Rows before: {initial_rows}, after: {len(df)}.")
    print(f"  Remaining nulls: {df.isnull().sum().sum()}\n")

    # ── Step 5: Perform exploratory analysis on repayment behavior ──
    print("Step 5: Performing EDA on repayment behavior...")

    # 5a – Recovery status distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='recovery_status', data=df, ax=ax, palette='viridis')
    ax.set_title('Distribution of Recovery Status')
    ax.set_xticklabels(['Not Recovered (0)', 'Recovered (1)'])
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'recovery_distribution.png'), dpi=150)
    plt.close(fig)

    # 5b – Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close(fig)

    # 5c – Repayment behavior plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(x='recovery_status', y='total_paid_amount', data=df, ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Total Paid Amount by Recovery Status')
    sns.boxplot(x='recovery_status', y='months_since_last_payment', data=df, ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Months Since Last Payment by Recovery Status')
    sns.boxplot(x='recovery_status', y='credit_score', data=df, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('Credit Score by Recovery Status')
    sns.boxplot(x='recovery_status', y='num_previous_defaults', data=df, ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Previous Defaults by Recovery Status')
    fig.suptitle('Repayment Behavior Analysis', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'repayment_behavior.png'), dpi=150)
    plt.close(fig)
    print("  Saved: recovery_distribution.png, correlation_heatmap.png, repayment_behavior.png\n")

    # ── Step 6: Encode categorical and scale numerical features ──
    le = LabelEncoder()
    df['employment_type'] = le.fit_transform(df['employment_type'])
    print(f"Step 6: Encoded employment_type -> {list(le.classes_)}")

    X = df.drop(columns=['recovery_status'])
    y = df['recovery_status']
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    print(f"  Scaled {len(feature_names)} numerical features with StandardScaler.\n")

    # ── Step 7: Handle class imbalance if present ──
    print(f"Step 7: Class distribution before SMOTE -> {dict(y.value_counts())}")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    print(f"  After SMOTE -> {dict(pd.Series(y_res).value_counts())}\n")

    # ── Step 8: Split dataset into training and testing sets ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )
    print(f"Step 8: Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

    # ── Step 9: Select suitable classification algorithm ──
    print("Step 9: Selected algorithms -> Logistic Regression, Random Forest, XGBoost\n")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
    }

    # ── Step 10 & 11: Train models, optimize recall and ROC-AUC ──
    best_model = None
    best_auc = 0
    best_model_name = ""
    roc_data = {}

    print("Step 10 & 11: Training and evaluating models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}\n")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

        # ── Step 12: Analyze confusion matrix results ──
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Recovered', 'Recovered'],
                    yticklabels=['Not Recovered', 'Recovered'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name} - Confusion Matrix')
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_cm.png'), dpi=150)
        plt.close(fig)

    # ROC curves on same plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Models')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'roc_curves.png'), dpi=150)
    plt.close(fig)

    print(f"[OK] Best Model: {best_model_name} (AUC = {best_auc:.4f})\n")

    # ── Step 13: Interpret recovery probability scores ──
    print("Step 13: Sample recovery probability predictions:")
    sample = X_test.iloc[:5]
    sample_proba = best_model.predict_proba(sample)[:, 1]
    sample_pred = best_model.predict(sample)
    for i in range(5):
        label = 'Recovered' if sample_pred[i] == 1 else 'Not Recovered'
        print(f"  Sample {i+1}: Probability = {sample_proba[i]:.4f} -> {label}")
    print()

    # ── Step 14: Save trained classification model ──
    joblib.dump(best_model, os.path.join(base_dir, 'loan_recovery_model.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(base_dir, 'label_encoder.pkl'))
    joblib.dump(feature_names, os.path.join(base_dir, 'feature_names.pkl'))
    print("Step 14: Saved model artifacts:")
    print("  -> loan_recovery_model.pkl")
    print("  -> scaler.pkl")
    print("  -> label_encoder.pkl")
    print("  -> feature_names.pkl\n")

    # ── Explainable AI: SHAP ──
    print("Generating SHAP summary plot...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    fig = plt.figure(figsize=(10, 6))
    # For binary classification, shap_values may be a list; use class 1
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  Saved: shap_summary.png\n")

    # ── Explainable AI: LIME ──
    print("Generating LIME explanation for a sample instance...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['Not Recovered', 'Recovered'],
        mode='classification'
    )
    exp = lime_explainer.explain_instance(
        data_row=np.array(X_test.iloc[0]),
        predict_fn=best_model.predict_proba
    )
    exp.save_to_file(os.path.join(plots_dir, 'lime_explanation.html'))
    print("  Saved: lime_explanation.html\n")

    print("=" * 60)
    print("Training pipeline complete. All assets saved successfully.")
    print("=" * 60)

if __name__ == '__main__':
    main()
