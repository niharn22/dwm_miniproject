import streamlit as st
import pandas as pd

# Function to load the dataset
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to display the dataset
def display_data(data):
    st.write("### Raw Data")
    st.write(data)

# Function for data cleaning
def clean_data(data):
    # Perform data cleaning operations here (replace NaN values, drop duplicates, etc.)
    # For example:
    cleaned_data = data.drop_duplicates()
    return cleaned_data

# Function for OLAP operations
def perform_olap(data):
    # Perform OLAP operations (GroupBy, aggregation, etc.)
    olap_data = data.groupby('Date').agg({
        'Open': 'mean',
        'High': 'max',
        'Low': 'min',
        'Close': 'mean',
        'Adj Close': 'mean',
        'Volume': 'sum'
    })
    return olap_data

# Main function
def main():
    st.title("Data Cleaning and OLAP Operations")

    # Upload the dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        display_data(data)

        # Data Cleaning
        st.header("Data Cleaning")
        cleaned_data = clean_data(data)
        st.write("### Cleaned Data")
        st.write(cleaned_data)

        # OLAP Operations
        st.header("OLAP Operations")
        olap_data = perform_olap(cleaned_data)
        st.write("### OLAP Result")
        st.write(olap_data)

if __name__ == "__main__":
    main()
