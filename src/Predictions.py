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
import gc  # For garbage collection

# === PERFORMANCE OPTIMIZATIONS ===
# 1. Cache the dataframe loading only 
@st.cache_data
def load_data():
    df = pd.read_csv("assets/main.csv")
    df = df[df['Customer Status'].isin(['Stayed', 'Churned'])]
    return df

# 2. Process data only when needed
@st.cache_data
def prepare_data(df):
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Monthly Charge' in features:
        features.remove('Monthly Charge')
    X = df[features]
    y = df['Customer Status'].map({'Stayed': 0, 'Churned': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to regular numpy arrays to prevent flags.writeable errors
    # These will be copied again in the train_and_evaluate_model function
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train) 
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

# === LIGHTWEIGHT PLOT FUNCTIONS ===
def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))  # Reduced size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(plt)
    plt.close()  # Explicitly close figure to free memory

def show_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)

def plot_feature_importance(model, features, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Only show top 10 features for large feature sets
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        
        # Only show top 10 features for better performance
        top_features = feature_df.head(10)
        
        plt.figure(figsize=(8, 5))  # Reduced size
        sns.barplot(x="Importance", y="Feature", data=top_features, color="skyblue")
        plt.title(f"Top 10 Features - {model_name}")
        st.pyplot(plt)
        plt.close()  # Explicitly close figure

# === OPTIMIZED TRAIN/EVAL FUNCTION ===
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, model_name, scale=False):
    # Create writeable copies of data to avoid the ValueError about writeable flag
    X_train_copy = np.array(X_train, copy=True)
    X_test_copy = np.array(X_test, copy=True)
    y_train_copy = np.array(y_train, copy=True)
    y_test_copy = np.array(y_test, copy=True)
    
    # Only scale if needed to save memory/computation
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_copy)
        X_test_scaled = scaler.transform(X_test_copy)
        model.fit(X_train_scaled, y_train_copy)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_copy, y_train_copy)
        y_pred = model.predict(X_test_copy)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"{model_name} Accuracy: {accuracy:.2f}")

    # Use tabs for model evaluation details to make UI cleaner
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])
    
    with eval_tab1:
        plot_confusion(y_test, y_pred, model_name)
    
    with eval_tab2:
        show_classification_report(y_test, y_pred)
    
    with eval_tab3:
        plot_feature_importance(model, X_train, feature_names, model_name)
    
    # Force garbage collection after training
    gc.collect()
    
    return model

# === STREAMLIT UI ===
st.markdown("""
    <h1 style='color: purple; font-size: 2.2em;'>
        📊 Machine Learning Models: Customer Churn Prediction 
    </h1>
""", unsafe_allow_html=True)

# Load data only when needed
df = load_data()

# Use a radio button instead of tabs to reduce initial rendering cost
model_choice = st.radio(
    "Select a model to train",
    ["Logistic Regression", "Random Forest", "XGBoost", "KNN", "Decision Tree", "PyCaret", "Mlflow"],
    horizontal=True
)

# Only prepare data when a model is going to be trained
if model_choice != "PyCaret" and model_choice != "Mlflow":
    with st.spinner("Preparing data..."):
        X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

# === MODEL TRAINING BASED ON SELECTION ===
if model_choice == "Logistic Regression":
    st.header("Logistic Regression")
    if st.button("Train Logistic Regression"):
        with st.spinner("Training Logistic Regression..."):
            model = LogisticRegression(max_iter=500)
            train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, "Logistic Regression", scale=True)

elif model_choice == "Random Forest":
    st.header("Random Forest Classifier")
    n_estimators = st.slider("Number of trees", 10, 100, 50)
    if st.button("Train Random Forest"):
        with st.spinner("Training Random Forest..."):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, "Random Forest")

elif model_choice == "XGBoost":
    st.header("XGBoost Classifier")
    if st.button("Train XGBoost"):
        with st.spinner("Training XGBoost..."):
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
            train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, "XGBoost")

elif model_choice == "KNN":
    st.header("K-Nearest Neighbors")
    k = st.slider("Choose K (Number of Neighbors)", min_value=1, max_value=20, value=5)
    if st.button("Train KNN"):
        with st.spinner("Training KNN..."):
            model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, f"KNN (K={k})", scale=True)

elif model_choice == "Decision Tree":
    st.header("Decision Tree Classifier")
    max_depth = st.slider("Select Max Depth", min_value=1, max_value=15, value=5)
    show_tree = st.checkbox("Show Decision Tree Visualization (may be slow)", value=False)
    
    if st.button("Train Decision Tree"):
        with st.spinner("Training Decision Tree..."):
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            trained_tree = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, f"Decision Tree (Max Depth={max_depth})")
            
            # Only generate the tree visualization if explicitly requested
            if show_tree:
                with st.spinner("Generating tree visualization (this may take a while)..."):
                    st.subheader("Decision Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))  # Reduced size
                    plot_tree(trained_tree, filled=True, feature_names=feature_names, 
                              class_names=['Stayed', 'Churned'], ax=ax, fontsize=8, max_depth=max_depth)
                    
                    # Use a controlled memory buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=100)  # Lower DPI
                    buf.seek(0)
                    st.image(buf, width=700)
                    plt.close(fig)
                    del buf  # Free memory
                    gc.collect()

elif model_choice == "PyCaret":
    st.header("PyCaret (AutoML)")
    st.write("PyCaret is a low-code machine learning library that allows quick model training and comparison.")
    try:
        # Load PyCaret results directly from file instead of computing
        ready_df = pd.read_csv('assets/Pycaret.csv')
        st.subheader("Model Comparison")
        st.dataframe(ready_df)
        st.success("Best model: **Gradient Boosting Classifier (gbc)**")
    except FileNotFoundError:
        st.error("PyCaret results file not found.")

elif model_choice == "Mlflow":
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

# Force garbage collection at the end
gc.collect()