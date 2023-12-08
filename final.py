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

# Function for binning
def perform_binning(data, column_to_bin, num_bins):
    st.header("Binning")

    # Select the column for binning
    column_data = data[column_to_bin]

    # Perform binning
    bins = pd.cut(column_data, bins=num_bins, labels=False)
    data['Bin'] = bins

    # Display binned data
    st.write("### Binned Data")
    st.write(data)

    # Plotting binned data
    st.subheader("Binned Data Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Bin'], kde=False)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.title("Binned Data Distribution")
    st.pyplot(plt)

# Function for K-Means clustering
def perform_clustering(data, num_clusters=3):
    st.header("K-Means Clustering")

    # Select columns for clustering
    columns_for_clustering = st.multiselect("Select Columns for Clustering", data.columns)
    if not columns_for_clustering:
        st.warning("Please select at least one column for clustering.")
        return

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

    # Plotting K-Means clusters (2D scatter plot for selected columns with clusters colored and centroids marked)
    st.subheader("K-Means Clusters (Scatter Plot)")
    scatter_columns = st.multiselect("Select columns for scatter plot", columns_for_clustering)
    if len(scatter_columns) >= 2:
        fig, ax = plt.subplots()
        scatter_plot = sns.scatterplot(data=data, x=scatter_columns[0], y=scatter_columns[1], hue='Cluster',
                                       palette='viridis', ax=ax)
        centroids = kmeans.cluster_centers_[:, :2]  # Take only the first two dimensions for 2D scatter plot
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red', label='Centroids')
        plt.xlabel(scatter_columns[0])
        plt.ylabel(scatter_columns[1])
        plt.legend()
        st.pyplot(fig)
    else:
        st.warning("Please select at least two columns for the scatter plot.")

# Function for data visualization
def visualize_data(data):
    st.header("Data Visualization")

    # Select columns for visualization
    columns_for_visualization = st.multiselect("Select Columns for Visualization", data.columns)

    if not columns_for_visualization:
        st.warning("Please select at least one column for visualization.")
        return

    # Line plot for selected columns over Date
    st.subheader("Line Plot over Time")
    fig, ax = plt.subplots()
    data.set_index('Date')[columns_for_visualization].plot(ax=ax)
    plt.xlabel("Date")
    plt.ylabel("Values")
    st.pyplot(fig)

    # Histogram of selected columns
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    data[columns_for_visualization].plot(kind='hist', bins=20, ax=ax)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    st.pyplot(fig)

    # Scatter plot for selected columns
    st.subheader("Scatter Plot")
    scatter_columns = st.multiselect("Select columns for scatter plot", columns_for_visualization)
    if len(scatter_columns) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=scatter_columns[0], y=scatter_columns[1], ax=ax)
        plt.xlabel(scatter_columns[0])
        plt.ylabel(scatter_columns[1])
        st.pyplot(fig)
    else:
        st.warning("Please select exactly two columns for the scatter plot.")

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

        # Data Visualization
        visualize_data(cleaned_data)

        # Clustering
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        perform_clustering(cleaned_data, num_clusters)

if __name__ == "__main__":
    main()

