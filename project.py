import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Function to load the dataset
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to display the dataset
def display_data(data):
    st.write("### Raw Data")
    st.write(data)

# Function for data cleaning and post-cleaning operations
def clean_data(data):
    # Perform data cleaning operations here (replace NaN values, drop duplicates, etc.)
    cleaned_data = data.drop_duplicates()
    
    # Handle missing values
    cleaned_data = cleaned_data.dropna()  # Drop rows with missing values
    
    # Calculate median and mode for numeric columns
    numeric_columns = cleaned_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    median_values = cleaned_data[numeric_columns].median()
    mode_values = cleaned_data[numeric_columns].mode().iloc[0]  # Select mode for each column
    
    return cleaned_data, median_values, mode_values

# Function to plot data
def plot_data(data):
    st.header("Data Visualization")

    # Line plot for Open and Close prices over Date
    st.subheader("Open and Close Prices over Time")
    fig, ax = plt.subplots()
    data.set_index('Date')[['Open', 'Close']].plot(ax=ax)
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)

    # Histogram of Volume
    st.subheader("Volume Distribution")
    fig, ax = plt.subplots()
    data['Volume'].plot(kind='hist', bins=20, ax=ax)
    plt.xlabel("Volume")
    plt.ylabel("Frequency")
    st.pyplot(fig)

    # Candlestick chart for stock prices
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    st.plotly_chart(fig)

    # Bar chart for average price comparison (excluding 'Adj Close')
    st.subheader("Average Price Comparison")
    price_columns = ['Open', 'High', 'Low', 'Close']
    avg_prices = data[price_columns].mean()
    fig, ax = plt.subplots()
    avg_prices.plot(kind='bar', ax=ax)
    plt.xlabel("Price Type")
    plt.ylabel("Average Price")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Scatter plot for High vs. Low prices
    st.subheader("High vs. Low Prices")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='High', y='Low', ax=ax)
    plt.xlabel("High Price")
    plt.ylabel("Low Price")
    st.pyplot(fig)

# Function for K-Means clustering
def perform_clustering(data, num_clusters=3):
    st.header("K-Means Clustering")

    # Select columns for clustering
    columns_for_clustering = ['Open', 'High', 'Low', 'Close', 'Volume']  # Adjusted columns list
    cluster_data = data[columns_for_clustering]

    # Handle missing values if any
    cluster_data.dropna(inplace=True)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(cluster_data)
    data['Cluster'] = cluster_labels

    # Display clustered data
    st.write("### Clustered Data")
    st.write(data)

    # Plotting K-Means clusters (2D scatter plot for High vs. Low prices with clusters colored)
    st.subheader("K-Means Clusters (High vs. Low Prices)")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='High', y='Low', hue='Cluster', palette='viridis')
    plt.xlabel("High Price")
    plt.ylabel("Low Price")
    plt.title("K-Means Clustering")
    plt.legend(title='Cluster')
    st.pyplot(plt)

# Main function
def main():
    st.title("DWM Mini Project - Data Cleaning, Visualization, and Clustering")

    # Upload the dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        display_data(data)

        # Data Cleaning
        st.header("Data Cleaning")
        cleaned_data, median_values, mode_values = clean_data(data)
        st.write("### Cleaned Data")
        st.write(cleaned_data)
        
        st.write("### Median Values")
        st.write(median_values)
        
        st.write("### Mode Values")
        st.write(mode_values)

        # Clustering
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        perform_clustering(cleaned_data, num_clusters)

        # Data Visualization
        plot_data(cleaned_data)

if __name__ == "__main__":
    main()
