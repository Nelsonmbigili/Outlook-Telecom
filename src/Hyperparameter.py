# import streamlit as st
# import mlflow
# import mlflow.sklearn
# import dagshub
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score


# # Example: Load your dataset and perform initial setup
# @st.cache_data
# def load_data():
#     df = pd.read_csv('assets/main.csv')  # Replace with actual dataset path
#     return df


# df = load_data()
# st.write("### Dataset Overview")
# st.dataframe(df.head())

# # Example: Hyperparameter tuning using MLFlow
# def log_experiment(model, model_name, hyperparameters, metrics):
#     with mlflow.start_run():
#         mlflow.log_params(hyperparameters)  # Log hyperparameters
#         mlflow.log_metrics(metrics)  # Log metrics like accuracy, precision, etc.
#         mlflow.sklearn.log_model(model, model_name)  # Log the trained model


# # --- Display Hyperparameter Tuning Experiences ---
# st.title("🎯 Hyperparameter Tuning & Best Model Selection")


# # Model selection
# model_choice = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Decision Tree"])


# # Hyperparameter selection
# if model_choice == "Decision Tree":
#     max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)
#     min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2)
#     hyperparameters = {"max_depth": max_depth, "min_samples_split": min_samples_split}
# elif model_choice == "KNN":
#     n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=15, value=5)
#     hyperparameters = {"n_neighbors": n_neighbors}
# elif model_choice == "Logistic Regression":
#     c_value = st.slider("Regularization Strength (C)", min_value=0.1, max_value=10.0, value=1.0)
#     hyperparameters = {"C": c_value}


# # Model training and logging
# if st.button("Train and Log Model"):
#     # Preprocess data
#     y = df['Customer Status']
#     X = df.drop(columns=['Customer Status'])
#     X = pd.get_dummies(X)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#     # Initialize model based on user choice
#     if model_choice == "Logistic Regression":
#         model = LogisticRegression(C=hyperparameters['C'])
#     elif model_choice == "KNN":
#         model = KNeighborsClassifier(n_neighbors=hyperparameters['n_neighbors'])
#     elif model_choice == "Decision Tree":
#         model = DecisionTreeClassifier(max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'])


#     # Train model
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)


#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     metrics = {"accuracy": accuracy}


#     # Log experiment in MLFlow
#     log_experiment(model, model_choice, hyperparameters, metrics)


#     # Display metrics and results
#     st.write(f"Model Accuracy: {accuracy:.2f}")


#     # Visualize Confusion Matrix
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     fig, ax = plt.subplots()
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_title('Confusion Matrix')
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('Actual')
#     st.pyplot(fig)


#     # Optionally, display model metrics
#     st.write("Classification Report:")
#     st.write(metrics)



