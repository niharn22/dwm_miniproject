import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

# Function for clustering
def perform_clustering(data, algorithm, columns, num_clusters, max_distance):
    # Extract selected columns for clustering
    X = data[columns]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == "K-Means":
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X_scaled)

    elif algorithm == "Hierarchical":
        # Apply hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        data['Cluster'] = hierarchical.fit_predict(X_scaled)

    elif algorithm == "DBSCAN":
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=max_distance)
        data['Cluster'] = dbscan.fit_predict(X_scaled)

    return data

# Function to visualize clustering results
def visualize_clusters(data, columns):
    # Scatter plot of the first two columns with color-coded clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[columns[0]], y=data[columns[1]], hue=data['Cluster'], palette='viridis')
    plt.title("Clustering Results")
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    st.pyplot(plt)

# Main function
def main():
    st.title("Clustering Analysis")

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

        # Clustering Analysis
        st.header("Clustering Analysis")

        # Choose columns for clustering
        clustering_columns = st.multiselect("Select Columns for Clustering", cleaned_data.columns)

        # Choose the clustering algorithm
        clustering_algorithm = st.selectbox("Choose Clustering Algorithm", ["K-Means", "Hierarchical", "DBSCAN"])

        if clustering_algorithm in ["K-Means", "Hierarchical"]:
            # Choose the number of clusters
            num_clusters = st.slider("Choose the Number of Clusters", min_value=2, max_value=10, value=3)
            max_distance = None
        else:
            # Choose the maximum distance for DBSCAN
            max_distance = st.slider("Choose the Maximum Distance for DBSCAN", min_value=0.1, max_value=2.0, value=0.5)
            num_clusters = None

        # Perform clustering
        clustered_data = perform_clustering(cleaned_data, clustering_algorithm, clustering_columns, num_clusters, max_distance)
        st.write("### Clustered Data")
        st.write(clustered_data)

        # Visualize clustering results
        visualize_clusters(clustered_data, clustering_columns)

if __name__ == "__main__":
    main()
