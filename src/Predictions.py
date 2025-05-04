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
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb

# === CACHED DATA LOADING AND SPLITTING ===
@st.cache_data
def load_and_split_data():
    df = pd.read_csv("assets/main.csv")
    df = df[df['Customer Status'].isin(['Stayed', 'Churned'])]  # filter classes
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Monthly Charge' in features:
        features.remove('Monthly Charge')
    X = df[features]
    y = df['Customer Status'].map({'Stayed': 0, 'Churned': 1})
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

# === PLOT UTILS ===
def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)

def show_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)

def plot_feature_importance(model, X, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_df, color="skyblue")
        plt.title(f"Feature Importance - {model_name}")
        st.pyplot(plt)
        plt.clf()

# === MAIN TRAIN/EVAL FUNCTION ===
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).copy()
        X_test = scaler.transform(X_test).copy()
    else:
        X_train = np.array(X_train, copy=True)
        X_test = np.array(X_test, copy=True)

    y_train = np.array(y_train, copy=True)
    y_test = np.array(y_test, copy=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # ✅ Display in Streamlit UI
    st.success(f"{model_name} Accuracy: {accuracy:.2f}")

    # Optionally show confusion matrix and classification report
    plot_confusion(y_test, y_pred, model_name)
    show_classification_report(y_test, y_pred)
    plot_feature_importance(model, pd.DataFrame(X_train, columns=feature_names), model_name)

    return model

# === LOAD DATA ONCE ===
(X_train, X_test, y_train, y_test), feature_names = load_and_split_data()

# === UI HEADER ===
st.markdown("""
    <h1 style='color: purple; font-size: 2.2em;'>
        📊 Machine Learning Models: Customer Churn Prediction
    </h1>
""", unsafe_allow_html=True)

# === TABS FOR MODELS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Logistic Regression", "Random Forest", "XGBoost", "KNN", "Decision Tree", "PyCaret", "Mlflow"
])

# === TAB 1: Logistic Regression ===
with tab1:
    st.header("Logistic Regression")
    if st.button("Train Logistic Regression"):
        model = LogisticRegression()
        train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "Logistic Regression", scale=True)

# === TAB 2: Random Forest ===
with tab2:
    st.header("Random Forest Classifier")
    if st.button("Train Random Forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "Random Forest")

# === TAB 3: XGBoost ===
with tab3:
    st.header("XGBoost Classifier")
    if st.button("Train XGBoost"):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        train_and_evaluate_model(model, X_train, X_test, y_train, y_test, "XGBoost")

# === TAB 4: KNN ===
with tab4:
    st.header("K-Nearest Neighbors")
    k = st.slider("Choose K (Number of Neighbors)", min_value=1, max_value=20, value=5)
    if st.button("Train KNN"):
        model = KNeighborsClassifier(n_neighbors=k)
        train_and_evaluate_model(model, X_train, X_test, y_train, y_test, f"KNN (K={k})", scale=True)

# === TAB 5: Decision Tree ===
with tab5:
    st.header("Decision Tree Classifier")
    max_depth = st.slider("Select Max Depth", min_value=1, max_value=15, value=5)
    if st.button("Train Decision Tree"):
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        trained_tree = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, f"Decision Tree (Max Depth={max_depth})")

        st.subheader("Full decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(20, 20)) 
        plot_tree(trained_tree, filled=True, feature_names=feature_names, class_names=['Stayed', 'Churned'], ax=ax, fontsize=12)
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

# === TAB 6: PyCaret ===
with tab6:
    st.header("PyCaret (AutoML)")
    st.write("PyCaret is a low-code machine learning library that allows quick model training and comparison.")
    try:
        ready_df = pd.read_csv('assets/Pycaret.csv')
        st.subheader("Model Comparison")
        st.dataframe(ready_df)
        st.success("Best model: **Gradient Boosting Classifier (gbc)**")
    except FileNotFoundError:
        st.error("PyCaret results file not found.")

# === TAB 7: Mlflow ===
with tab7:
    st.header("Hyperparameter Tuning (Mlflow)")
    st.write("""
        We trained several models using different hyperparameters like K in KNN and max_depth in Decision Trees.
        Use the button below to view the experiment tracking report on DagsHub.
    """)
    dagshub_url = "https://dagshub.com/Nelsonmbigili/OutLook.mlflow/#/experiments/0"
    st.markdown(f"""
        <a href="{dagshub_url}" target="_blank">
            <button style="padding:10px 20px;font-size:16px;border:none;border-radius:5px;background-color:#4CAF50;color:white;cursor:pointer;">
                🔍 View Full MLflow Report on DagsHub
            </button>
        </a>
    """, unsafe_allow_html=True)
