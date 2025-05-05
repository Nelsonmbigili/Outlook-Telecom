import streamlit as st

st.markdown(
    """
    <div style='white-space: nowrap; overflow-x: auto;'>
        <h1 style='color: purple; display: inline-block; font-size: 2.2em; margin: 0;'>
           Conclusion & Recommendations for OutLook Telecom 🛜
        </h1>
    </div>
    """,
    unsafe_allow_html=True)
st.subheader("Summary of Findings")

st.write("""
Our analysis of OutLook Telecom's 2024 California dataset provided valuable insights into customer behavior, churn patterns, and revenue trends. 

Some of the key findings includes the following: 
- **Customer Demographics**: The majority of customers are in the 30-39 age range while the Average age of customers is appx 47 years with ~47% having dependents.

""")

st.subheader("Actionable Recommendations")
st.write("""
Retention Strategies
- Personalized Offers for Short-Tenure Customers: Target customers with shorter tenures by offering tailored promotions, improving their loyalty and retention.
- Referral Incentives: Introduce programs that reward customers for referring others, helping both with customer acquisition and increasing retention rates.
- Fiber Optic Service Enhancements: Increase the perceived value of fiber optic services by offering bundled discounts or improving speed offerings to incentivize longer-term subscriptions.

Contract & Billing Optimization
- Long-Term Contracts: Offer discounts or special promotions for customers committing to 1- or 2-year contracts, encouraging longer retention and stability.
- Paperless Billing & Auto-Pay Promotion: Encourage customers to switch to paperless billing and auto-pay systems, minimizing friction and ensuring a smoother payment process.

Competitive Pricing & Service Enhancements
- Monitor Competitor Pricing: Regularly evaluate competitor pricing to ensure your plans remain competitive, adjusting accordingly to attract and retain customers.
- Enhanced Customer Support: Focus on strengthening customer support, especially for high-churn segments, providing personalized, high-quality service to reduce churn.

Data-Driven Decision Making
- Expand Predictive Modeling: Implement predictive models to forecast customer churn in real time, allowing for proactive retention efforts.
- Leverage Geographic Insights: Use data to identify high-churn zip codes or regions and launch targeted campaigns to address specific local issues, improving retention.
""")

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

st.success("Thank you for exploring the OutLook Telecom Annual Data Review app!")