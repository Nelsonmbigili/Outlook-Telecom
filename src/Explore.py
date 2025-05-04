# import streamlit as st
# import pandas as pd
# import numpy as np
# import codecs
# import streamlit.components.v1 as components
# from streamlit_option_menu import option_menu

# st.markdown(
#     """
#     <div style='white-space: nowrap; overflow-x: auto;'>
#         <h1 style='color: purple; display: inline-block; font-size: 2.2em; margin: 0;'>
#             OutLook Telecom: Annual Data Review
#         </h1>
#     </div>
#     """,
#     unsafe_allow_html=True)
# st.subheader(":violet[Data Overview]")

# dataset = pd.read_csv("assets/main.csv")
# dataset_dictionary = pd.read_csv("assets/telecom_data_dictionary.csv", encoding='latin1')

# selected = option_menu(menu_title=None, options=["01: Dictionary", "02: Summary",], orientation="horizontal")

# if(selected=="01: Dictionary"):
#     st.subheader(":violet[Dataset]")
#     st.markdown("""
#     **The dataset we shall explore contains records from OutLook Telecom**,  
#     a company that provides phone and internet services to **7,043 customers across California**.  

#     It includes comprehensive details about **customer demographics**,  
#     **service subscriptions**, **account status**, and whether or not the customer has churned.  

#     This information allows us to:  
#     - Identify **behavioral trends**,  
#     - Understand the **drivers of customer churn**, and  
#     - Provide **data-driven recommendations** to support OutLook Telecom’s customer retention and business growth strategies.
#     """)
#     st.subheader(":violet[Columns & Descriptions]")
#     dataset_columns = pd.DataFrame({'Columns': dataset.columns})
#     st.dataframe(dataset_columns, width=700)

#     field_options = dataset_dictionary['Field'].unique()  
#     selected_field = st.selectbox("Select a field to view its description:", field_options)
#     selected_row = dataset_dictionary[dataset_dictionary['Field'] == selected_field].iloc[0]
#     st.markdown('<span style="color:blue; font-weight:bold;">Description</span>', unsafe_allow_html=True)
#     st.code(f'"{selected_row["Description"]}"')

# elif(selected=="02: Summary"):
#     # Display Dataset Sample
#     st.write("### Data Preview: 10 rows")
#     view_option = st.radio("View from:", ("Top", "Bottom"))
#     if view_option == "Top":
#         st.dataframe(dataset.head(10))
#     else:
#         st.dataframe(dataset.tail(10))

#     # Shape of the dataset
#     st.success(f"**Dataset Shape:** {dataset.shape[0]} rows and {dataset.shape[1]} columns")

#     # Missing values
#     st.write("### Missing Values")
#     missing = dataset.isnull().sum()
#     missing = missing[missing > 0]
#     if not missing.empty:
#         missing_df = pd.DataFrame({
#             'Column Name': missing.index,
#             'Missing Values Count': missing.values
#         })
#         st.write(missing_df)
#     else:
#         st.write("No missing values found.")

#     # Basic statistics
#     st.write("### Statistical Summary")
#     st.write(dataset.describe())

#     # Data types
#     st.write("### Data Types")
#     dtypes = dataset.dtypes
#     dtype_details = {}
#     for dtype in dtypes.unique():
#         columns = dtypes[dtypes == dtype].index.tolist()
#         dtype_details[str(dtype)] = {
#             "Columns": ", ".join(columns),
#             "Count": len(columns)
#         }

#     dtype_df = pd.DataFrame(dtype_details).T.reset_index()
#     dtype_df.columns = ['Data Type', 'Columns', 'Count']
#     st.write(dtype_df)