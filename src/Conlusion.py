import streamlit as st

st.markdown(
    """
    <div style='white-space: nowrap; overflow-x: auto;'>
        <h1 style='color: purple; display: inline-block; font-size: 2.2em; margin: 0;'>
           📊 Project Conclusion: Telecom Customer Churn Prediction
        </h1>
    </div>
    """,
    unsafe_allow_html=True)
st.subheader("Business Advisory Based on Our Churn Prediction Results📉")

st.write("""
Through our comprehensive analysis of the telecom dataset, we discovered that **certain customer behaviors and contract attributes are strongly linked with customer churn**. 

- **Contract Type**: Customers on *month-to-month contracts* are significantly more likely to churn compared to those on longer-term contracts.
- **Monthly Charges**: Higher charges correlate with increased churn, indicating sensitivity to pricing.
- **Tenure**: Customers with a short tenure are more likely to leave, possibly due to poor onboarding or unmet expectations.
""")

st.subheader("Model Performance Summary 🔍 ")

# Display model performance metrics
st.markdown("""
| Model                        | Accuracy | Precision (Churn) | Recall (Churn) | F1-score (Churn) |
|-----------------------------|----------|-------------------|----------------|------------------|
| Logistic Regression         | **0.8035** | 0.8006            | 0.8035         | 0.8018           |
| K-Nearest Neighbors (N=5)   | **0.7656** | 0.7601            | 0.7656         | 0.7623           |
| Decision Tree (depth=5)     | **0.8194** | 0.8377            | 0.8194         | 0.7944           |
| Random Forest               | **0.8513** | 0.8509            | 0.8513         | 0.8426           |
| XGBoost                     | **0.8467** | 0.8427            | 0.8467         | 0.8420           |
""", unsafe_allow_html=True)

st.write("**Decision Tree performed best** in both accuracy and interpretability, making it ideal for churn risk analysis.")

st.subheader("Strategic Recommendations")

st.markdown("""
- **Bundle longer-term contracts**: Offer incentives for yearly plans to improve retention.
- **Target high-risk groups**: Use churn likelihood scores to trigger retention actions for customers with high bills and short tenure.
- **Simplify pricing models**: Address churn causes rooted in unclear or high monthly charges.
""")

st.subheader("Improvements for Future Iterations 🔧")

st.markdown("""
-  **Expand dataset diversity**: Include customers from other states and consider data from competitors.
-  **Add behavioral data**: Incorporate usage patterns like call/data usage and customer support history.
-  **Enable real-time ML**: Integrate prediction systems into live dashboards or alerts.
-  **Improve model explainability**: Use SHAP or LIME to provide transparent model insights to business teams.
""")

st.success("This project successfully demonstrates the application of machine learning models to predict and reduce customer churn in telecom services, guiding actionable strategies based on data insights.")