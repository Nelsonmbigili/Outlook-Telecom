import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from streamlit_option_menu import option_menu

st.markdown(
    """
    <div style='white-space: nowrap; overflow-x: auto;'>
        <h1 style='color: purple; display: inline-block; font-size: 2.2em; margin: 0;'>
            OutLook Telecom: Annual Data Review
        </h1>
    </div>
    """,
    unsafe_allow_html=True)
st.subheader(":violet[Data Visualization]")

dataset = pd.read_csv("assets/main.csv")
dataset_cleaned = dataset.dropna(subset=['Gender', 'Churn Category', 'Tenure in Months', 'Monthly Charge', 'Total Charges'])

# Horizontal navigation for pages
selected_page = option_menu(
    menu_title=None, 
    options=["Demographics", "Services", "Financial Trends"],
    orientation="horizontal",
    icons=['house', 'person', 'bar-chart'],
    default_index=0
)

# Page 1: Customer Demographics
if selected_page == "Demographics":
    st.subheader(":violet[Customer Demographics]")
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    labels = ["11–19", "20–29","30–39",  "40–49",  "50–59",  "60–69", "70–79","80+"]
    dataset["Age Range"] = pd.cut(dataset["Age"], bins=bins, labels=labels, right=False)

    # Distribution selector
    distribution_type = st.selectbox("Select Distribution Type",["Gender", "Age Range"])

    # Distribution by Gender
    if distribution_type == "Gender":
        gender_counts = dataset['Gender'].value_counts()
        st.subheader(":violet[Customer Distribution by Gender]")
        fig, ax = plt.subplots()
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, 
        colors=sns.color_palette("deep", len(gender_counts)))
        ax.axis('equal')  
        st.pyplot(fig)

    # Distribution by Age Range
    elif distribution_type == "Age Range":
        age_counts = dataset["Age Range"].value_counts().sort_index()
        st.subheader(":violet[Customer Distribution by Age Range]")
        fig, ax = plt.subplots()
        sns.barplot(x=age_counts.index, y=age_counts.values, palette="colorblind", ax=ax)
        ax.set_ylabel("Number of Customers")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlabel("Age Range")
        st.pyplot(fig)
    
    # Map of Customer Distribution In California
    st.subheader(":violet[Map of Customer Distribution In California]")
    lat = dataset["Latitude"]
    lon = dataset["Longitude"]
    data = pd.DataFrame({'lat': lat, 'lon': lon})
    st.map(data)


    city_counts = dataset['City'].value_counts().reset_index()
    city_counts.columns = ['City', 'Customer Count']

    # Most Popular Cities
    st.subheader(":violet[Top Cities by Customer Count]")
    sorted_data = city_counts.sort_values(by='Customer Count', ascending=False)
    top_data = sorted_data.head(5)  
    fig, ax = plt.subplots()
    sns.barplot(x='Customer Count', y='City', data=top_data, palette='viridis', ax=ax)
    ax.set_xlabel("Customer Count")
    ax.set_ylabel("City")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Least Popular Cities
    st.subheader(":violet[Least Popular Cities by Customer Count]")
    sorted_data = city_counts.sort_values(by='Customer Count', ascending=True)
    bottom_data = sorted_data.head(10)
    st.dataframe(bottom_data[['City', 'Customer Count']] ,width=700)


# Page 2: Service and Subscription
if selected_page == "Services":
    st.subheader(":violet[Services and Subscriptions]")
    service_options = st.selectbox("Select a Service Attribute", [
        "Phone Service", "Internet Service", "Online Security", "Online Backup",
        "Device Protection Plan", "Premium Tech Support", "Streaming TV",
        "Streaming Movies", "Streaming Music", "Unlimited Data"
    ])

    st.subheader(f":violet[Customer Distribution by {service_options}]")

    service_counts = dataset_cleaned[service_options].value_counts().reset_index()
    service_counts.columns = [service_options, "Count"]

    fig, ax = plt.subplots()
    sns.barplot(x=service_options, y="Count", data=service_counts, palette="pastel", ax=ax)
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel(service_options)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Correlation between Internet Type and Selected Service
    st.subheader(":violet[Internet Type vs Selected Service]")

    internet_service = st.selectbox(
        "Select a Service for Internet Type Comparison",
        options=[
            "Streaming TV", "Streaming Movies", "Streaming Music",
            "Device Protection Plan", "Online Security", "Online Backup",
            "Premium Tech Support", "Unlimited Data", "Phone Service"
        ],
        index=0
    )

    fig2 = px.histogram(
        dataset_cleaned,
        x="Internet Type",
        color=internet_service,
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(
        xaxis_title="Internet Type",
        yaxis_title="Customer Count",
        title={
        'text': f"Internet Type vs {internet_service} Usage",
        'x': 0.5, 
        'xanchor': 'center' 
    },) 
   
    st.plotly_chart(fig2, use_container_width=True)


    # Correlation between Contract Type and Selected Service
    st.subheader(":violet[Contract Type vs Selected Service]")

    contract_service = st.selectbox(
        "Select a Service for Contract Type Comparison",
        options=[
            "Streaming TV", "Streaming Movies", "Streaming Music",
            "Device Protection Plan", "Online Security", "Online Backup",
            "Premium Tech Support", "Unlimited Data", "Phone Service"
        ],
        index=0,
        key="contract_service"  # To avoid conflict with previous selectbox
    )

    fig3 = px.histogram(
        dataset_cleaned,
        x="Contract",
        color=contract_service,
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig3.update_layout(
        xaxis_title="Contract Type",
        yaxis_title="Customer Count",
        title={
        'text': f"Contract Type vs {contract_service} Usage",
        'x': 0.5,
        'xanchor': 'center'
    },
        )
    st.plotly_chart(fig3, use_container_width=True)
# Page 3: Financial Trends and Customer Lifetime
if selected_page == "Financial Trends":
    st.subheader(":violet[Financial Trends]")

    import numpy as np
    import pandas as pd
    import plotly.express as px

    # -------------------------
    # Setup financial data
    # -------------------------
    financial_df = dataset_cleaned.copy()
    financial_df['Revenue'] = financial_df['Monthly Charge'] * financial_df['Tenure in Months']
    financial_df['Churned'] = financial_df['Churn Category'].apply(lambda x: 0 if x == "No Churn" else 1)


    st.subheader(":violet[1. Revenue Lost by Churn Category]")

    churned = financial_df[financial_df["Churn Category"] != "No Churn"]
    churn_loss = churned.groupby("Churn Category")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)

    fig1 = px.bar(
        churn_loss,
        x="Churn Category",
        y="Revenue",
        color="Churn Category",
        title="Total Revenue Lost by Churn Reason",
        text_auto='.2s'
    )
    fig1.update_layout(
        yaxis_title="Lost Revenue (USD)",
        yaxis_tickprefix="$",
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(":violet[2. Churn Rate by Churn Category]")


    churn_only = financial_df[financial_df["Churn Category"] != "No Churn"]

    churn_cats = churn_only["Churn Category"].value_counts(normalize=True).reset_index()
    churn_cats.columns = ["Churn Category", "Churn Rate"]
    churn_cats["Churn Rate (%)"] = churn_cats["Churn Rate"] * 100

    fig2 = px.bar(
        churn_cats,
        x="Churn Category",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
        color="Churn Category",
        title="Churn Breakdown by Category"
    )
    fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # 3. Average Revenue by Service Used
    # -------------------------
    st.subheader(":violet[3. Average Revenue by Service Used]")

    service_columns = [
        "Phone Service", "Internet Service", "Online Security", "Online Backup",
        "Device Protection Plan", "Premium Tech Support", "Streaming TV",
        "Streaming Movies", "Streaming Music", "Unlimited Data"
    ]

    avg_rev = []
    for service in service_columns:
        if service in financial_df.columns:
            rev = financial_df[financial_df[service] == "Yes"]['Revenue'].mean()
            avg_rev.append((service, rev))

    service_df = pd.DataFrame(avg_rev, columns=["Service", "Avg Revenue"]).sort_values(by="Avg Revenue", ascending=False)

    fig3 = px.bar(
        service_df,
        x="Service",
        y="Avg Revenue",
        title="Average Revenue by Service",
        color="Service",
        text_auto='.2s'
    )
    fig3.update_layout(yaxis_title="Avg Revenue (USD)", yaxis_tickprefix="$", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)