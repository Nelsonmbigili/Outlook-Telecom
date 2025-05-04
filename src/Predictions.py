import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import joblib

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
@st.cache_data
def train_and_evaluate_model_cached(model, X_train, X_test, y_train, y_test, model_name, scale=False):
    # Optionally scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = np.array(X_train, copy=True)
        X_test = np.array(X_test, copy=True)

    # Ensure labels are writable arrays
    y_train = np.array(y_train, copy=True)
    y_test = np.array(y_test, copy=True)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    
    return model

# === LOAD DATA ONCE ===
(X_train, X_test, y_train, y_test), feature_names = load_and_split_data()

# === UI HEADER ===
st.markdown("""
    <h1 style='color: purple; font-size: 2.2em;'>
        📊 Machine Learning Models: Customer Churn Prediction
    </h1>
""", unsafe_allow_html=True)

# === CACHED MODEL LOADING ===
if 'decision_tree_model' not in st.session_state:
    st.session_state.decision_tree_model = None  # Initialize if it doesn't exist
if 'knn_model' not in st.session_state:
    st.session_state.knn_model = None  # Initialize if it doesn't exist

# === TABS FOR MODELS ===
tab1, tab2 = st.tabs([
    "Decision Tree", "KNN"
])

# === TAB 1: Decision Tree ===
with tab1:
    st.header("Decision Tree Classifier")
    max_depth = st.slider("Select Max Depth", min_value=1, max_value=15, value=5)
    if st.button("Train Decision Tree"):
        with st.spinner('Training your model...'):
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            trained_tree = train_and_evaluate_model_cached(model, X_train, X_test, y_train, y_test, f"Decision Tree (Max Depth={max_depth})")

            # Store the trained model in session_state
            st.session_state.decision_tree_model = trained_tree
            st.success("Model trained successfully!")
            # Save model for future use
            joblib.dump(trained_tree, 'decision_tree_model.pkl')
    
    # Show the model evaluation
    if st.session_state.decision_tree_model is not None:
        trained_model = st.session_state.decision_tree_model
        y_pred = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        plot_confusion(y_test, y_pred, "Decision Tree")

        # Show feature importance
        plot_feature_importance(trained_model, X_train, "Decision Tree")

        # Optionally, show decision tree visualization
        if st.checkbox("Show full decision tree visualization"):
            fig, ax = plt.subplots(figsize=(20, 20)) 
            plot_tree(trained_model, filled=True, feature_names=feature_names, class_names=['Stayed', 'Churned'], ax=ax, fontsize=12)
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

# === TAB 2: KNN ===
with tab2:
    st.header("K-Nearest Neighbors (KNN)")
    n_neighbors = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=15, value=5)
    if st.button("Train KNN"):
        with st.spinner('Training your model...'):
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            trained_knn = train_and_evaluate_model_cached(model, X_train, X_test, y_train, y_test, f"KNN (K={n_neighbors})")

            # Store the trained model in session_state
            st.session_state.knn_model = trained_knn
            st.success("Model trained successfully!")
            # Save model for future use
            joblib.dump(trained_knn, 'knn_model.pkl')
    
    # Show the model evaluation
    if st.session_state.knn_model is not None:
        trained_model = st.session_state.knn_model
        y_pred = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        plot_confusion(y_test, y_pred, "KNN")

        # Optionally show classification report
        show_classification_report(y_test, y_pred)