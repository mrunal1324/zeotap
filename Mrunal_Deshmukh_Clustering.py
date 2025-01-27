import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data():
    """Load all required datasets and convert date columns to datetime."""
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')
    
    return customers_df, products_df, transactions_df

def prepare_advanced_features(customers_df, transactions_df, products_df):
    """Prepare comprehensive features for clustering."""
    # RFM Features
    latest_date = transactions_df['TransactionDate'].max()
    
    rfm = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (latest_date - x.max()).days,  # Recency
        'TransactionID': 'count',  # Frequency
        'TotalValue': 'sum'  # Monetary
    }).rename(columns={
        'TransactionDate': 'Recency',
        'TransactionID': 'Frequency',
        'TotalValue': 'Monetary'
    })
    
    # Product Category Preferences
    category_preferences = transactions_df.merge(products_df, on='ProductID')\
        .pivot_table(
            index='CustomerID',
            columns='Category',
            values='TotalValue',
            aggfunc='sum',
            fill_value=0
        )
    
    # Combine RFM and Category Preferences
    features = pd.concat([rfm, category_preferences], axis=1)
    
    # Handle missing values
    features = features.fillna(0)
    
    return features

def evaluate_clusters(features, max_clusters=10):
    """Evaluate different numbers of clusters using DB Index and Silhouette Score."""
    results = []
    
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(features)
        
        db_score = davies_bouldin_score(features, labels)
        silhouette = silhouette_score(features, labels)
        
        results.append({
            'n_clusters': n,
            'db_score': db_score,
            'silhouette_score': silhouette
        })
    
    return pd.DataFrame(results)

def visualize_clusters(features, labels, save_path='cluster_visualization.png'):
    """Create visualizations of clusters using PCA for dimensionality reduction."""
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of clusters
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Customer Segments (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(save_path)
    plt.close()

def main():
    # Load data
    customers_df, products_df, transactions_df = load_data()
    
    # Prepare features
    features = prepare_advanced_features(customers_df, transactions_df, products_df)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering with KMeans
    optimal_n_clusters = 7  # You can adjust this based on your evaluation
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Calculate Silhouette Score and Davies-Bouldin Index
    silhouette = silhouette_score(features_scaled, clusters)
    db_index = davies_bouldin_score(features_scaled, clusters)
    
    # Calculate cluster sizes
    cluster_sizes = np.bincount(clusters).tolist()
    
    # Save results in the specified format
    with open('Mrunal_Deshmukh_Clustering.txt', 'w') as f:
        f.write("Clustering Results:\n\n")
        f.write(f"n_clusters: {optimal_n_clusters}\n")
        f.write(f"db_index: {db_index}\n")
        f.write(f"silhouette_score: {silhouette}\n")
        f.write(f"cluster_sizes: {cluster_sizes}\n")
    
    print("Clustering completed and results saved to 'Mrunal_Deshmukh_Clustering.txt'.")

if __name__ == "__main__":
    main() 