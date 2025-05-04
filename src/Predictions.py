import streamlit as st
import pandas as pd
import io
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.tree import plot_tree

# Load Data
df = pd.read_csv("assets/main.csv")
features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Monthly Charge' in features:
    features.remove('Monthly Charge')

target = 'Customer Status'
df = df[df[target].isin(['Stayed', 'Churned'])] 
X = df[features]
y = df[target].map({'Stayed': 0, 'Churned': 1}) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot confusion matrix
def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)

# Show classification report
def show_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)

# Ploting feature importance
def plot_feature_importance(model, X, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        })
        feature_df = feature_df.sort_values("Importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_df, color="skyblue")
        plt.title(f"Feature Importance - {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        st.pyplot(plt)
        plt.clf()

# Training and evaluating models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"**Accuracy:** {round(acc, 4)}")
    show_classification_report(y_test, y_pred)
    plot_confusion(y_test, y_pred, model_name)
    plot_feature_importance(model, X, model_name)

st.markdown(
    """
    <div style='white-space: nowrap; overflow-x: auto;'>
        <h1 style='color: purple; display: inline-block; font-size: 2.2em; margin: 0;'>
           📊 Machine Learning Models: Customer Churn Prediction
        </h1>
    </div>
    """,
    unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Logistic Regression", "Random Forest", "XGBoost", "KNN", "Decision Tree", "PyCaret", "Mlflow"])

# Logistic Regression
with tab1:
    st.header("Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")

# Random Forest
with tab2:
    st.header("Random Forest Classifier")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "Random Forest")

# XGBoost
with tab3:
    st.header("XGBoost Classifier")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "XGBoost")

# K-Nearest Neighbors
with tab4:
    st.header("K-Nearest Neighbors")
    k = st.slider("Choose K (Number of Neighbors)", min_value=1, max_value=20, value=5)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=k)
    train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, f"KNN (K={k})")

# Decision Tree
with tab5:
    st.header("Decision Tree Classifier")
    max_depth = st.slider("Select Max Depth", min_value=1, max_value=15, value=5)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    train_and_evaluate_model(model, X_train, X_test, y_train, y_test, f"Decision Tree (Max Depth={max_depth})")

    # Decision Tree Visualization
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 20)) 
    plot_tree(model, filled=True, feature_names=X.columns, class_names=['Stayed', 'Churned'], ax=ax, fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    html_code = f"""
    <div style=" overflow-x: auto;">
        <img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" width="100%">
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    plt.close(fig) 

# PyCaret
with tab6:
    st.header("PyCaret (AutoML)")
    st.write("PyCaret is a low-code machine learning library that allows quick model training and comparison.")
    st.success("✅ PyCaret Report completed.")
    ready_df = pd.read_csv('assets/Pycaret.csv')
    best_model = "Gradient Boosting Classifier (gbc)"

    st.subheader("Model Comparison")
    st.dataframe(ready_df)
    st.success(f"Best model: **{best_model}**")

with tab7:
    st.header("Hyperparameter Tunning (Mlflow))")
    st.write("""
        We trained several models using different values for key hyperparameters like the number of neighbors (for KNN) and the max depth (for Decision Trees).
        We used Mlflow to track each experiments. Use the buton below to view report.
        """)
    dagshub_url = "https://dagshub.com/Nelsonmbigili/OutLook.mlflow/#/experiments/0"

    dagshub_url = "https://dagshub.com/Nelsonmbigili/OutLook.mlflow/#/experiments/0"
    st.markdown(f"""
        <a href="{dagshub_url}" target="_blank">
            <button style="padding:10px 20px;font-size:16px;border:none;border-radius:5px;background-color:#4CAF50;color:white;cursor:pointer;">
                🔍 View Full MLflow Report on DagsHub
            </button>
        </a>
    """, unsafe_allow_html=True)
    