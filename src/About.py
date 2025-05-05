import streamlit as st
from PIL import Image

# Loading Animation
with st.spinner('Loading page...'):
    # Page Title 
    st.markdown(
    """
    <div style='white-space: nowrap; overflow-x: auto;'>
        <h1 style='color: purple; display: inline-block; font-size: 2.2em; margin: 0;'>
            OutLook Telecom: Annual Data Review
        </h1>
    </div>
    """,
    unsafe_allow_html=True)

    # Logo Image
    image_path = Image.open("assets/OutLook.png") 
    st.image(image_path, use_container_width=True)

    # About Section
    # About Section
st.subheader(":violet[About]")
st.markdown("""
**OutLook Telecom: Annual Data Review** app is designed to provide insights and predictions based on the data collected over the past year by OutLook Telecom. This platform leverages predictive analytics and advanced data visualization techniques to help stakeholders make informed decisions about business strategies and opportunities for growth.
            Our primary Focus is to analyse Customer retention and churn, which are critical factors in the telecom due to competition in the industry.
""")

st.subheader(":violet[Purpose]")
st.markdown("""
The telecom industry generates a wealth of data, but extracting meaningful insights from this data can be challenging. By building this platform for **Outlook Telecom** using their 2024 dataset from the state of **Califonia**, we aim to provide decision-makers with clear, actionable insights to help drive strategic initiatives, improve customer experiences, and optimize business performance and most imporntantly **Retain Customers**.
""")

st.subheader(":violet[Usability]")
st.markdown("""
- **Data Exploration**: A thorough examination of the dataset used, including available columns, context, statistical summary, and data types.
- **Data Visualization**: Interactive charts and graphs that provide a clear visual representation of the data, categorised into Customer Demographics and Provided services and Financial trends.  
- **Predictive Analytics**: Predictive Machine Learning models that forecast trends of Customer status or Customer Retention within Outlook Telecom.  
- **Actionable Insights**: Data-driven suggestions for improving business operations and customer satisfaction and retention.
""")

st.subheader(":violet[Objectives]")
st.markdown("""
Through this app, we hope to empower OutLook Telecom's management with the tools they need to analyze performance, identify areas for improvement, and make data-driven decisions that foster growth and success in the telecom industry.
""")

st.subheader(":violet[Key Features]")
st.markdown("""
- **Welcome Page**: An introduction to the app, our client OutLook Telecom, and the the team
- **About Page**: A brief overview of the app's purpose and objectives, Features and Usability.  
- **Explore Page**: A detailed summary of the dataset, including available columns.
- **Predictions Page**: Machine learning models predicting customer churn and compare model perfomance.
- **Conclusion Page**: A summary of the findings and recommendations based on the analysis.
                 
We hope you find the platform insightful and beneficial in making strategic decisions for OutLook Telecom's growth.
""")

st.success("Dive into the data, discover insights, and keep learning .")