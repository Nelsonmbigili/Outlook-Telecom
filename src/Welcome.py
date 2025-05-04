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

    st.subheader(":violet[2024 Insights — Empowering Business Decisions]")

    # Welcome page text
    st.markdown("""
    ---
    ## 👋 Welcome!
    After the success of our [**FifaDataLab**](https://the-fifa-data-lab.streamlit.app/) app, our team was approached by  [**OutLook Telecom**](https://docs.google.com/document/d/1mJeo8kqCZ34xeaLnzFnoai7v7LieYiSjvxfZ39FyzzU/edit?usp=sharing) to analyze their data over the past year and provide actionable insights.
    
    Leveraging our experience in predictive analytics, data analysis and visualization, we’ve built this interactive platform to:
    - Explore key patterns and performance metrics in their data.
    - Present insights that highlight areas of opportunity.
    - Suggest evidence-based measures to guide strategic decision-making.
    
    """)

    # Team Members
    st.write("### Meet the Team:")
    team_members = {
        "Nelson Mbigili": "nfm8340@nyu.edu",
        "Ryan Opande": "rjo9414@nyu.edu",
        "Dhruv Gopan": "dag10005@nyu.edu",
        "Sean Shapiro": "sms10116@nyu.edu"
    }
    for name, email in team_members.items():
        st.write(f"- **{name}** ({email})")

    st.success("Made with ❤️ =>  Dive into the data, discover insights, and keep learning .")
