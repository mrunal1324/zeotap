import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    """Load all required datasets and convert date columns to datetime."""
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')
    
    return customers_df, products_df, transactions_df

def create_advanced_features(customers_df, transactions_df, products_df):
    """Create comprehensive feature matrix for customers."""
    # Basic transaction features
    transaction_features = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'std', 'count'],
        'Quantity': ['sum', 'mean', 'std'],
    })
    
    # Flatten MultiIndex columns
    transaction_features.columns = ['_'.join(col).strip() for col in transaction_features.columns.values]
    transaction_features = transaction_features.reset_index()
    
    # Recency Feature
    latest_date = transactions_df['TransactionDate'].max()
    customer_recency = transactions_df.groupby('CustomerID')['TransactionDate'].agg(
        lambda x: (latest_date - x.max()).days
    ).reset_index().rename(columns={'TransactionDate': 'Recency'})
    
    # Merge transaction features with recency
    customer_features = pd.merge(transaction_features, customer_recency, on='CustomerID', how='left')
    
    # Handle missing recency values (if any)
    customer_features['Recency'] = customer_features['Recency'].fillna((latest_date - transactions_df['TransactionDate'].min()).days)
    
    # Product Category Preferences
    category_preferences = transactions_df.merge(products_df, on='ProductID')
    category_pivot = category_preferences.pivot_table(
        index='CustomerID',
        columns='Category',
        values='TotalValue',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Combine all features
    customer_features = pd.merge(customer_features, category_pivot, on='CustomerID', how='left')
    
    # Fill any remaining missing values with 0
    customer_features = customer_features.fillna(0)
    
    return customer_features

def calculate_similarity_scores(features_scaled):
    """Calculate similarity using cosine similarity."""
    similarity_matrix = cosine_similarity(features_scaled)
    return similarity_matrix

def find_lookalikes(customer_id, customer_features, similarity_matrix, n_recommendations=3):
    """Find similar customers based on similarity scores."""
    # Get the index of the customer_id
    customer_idx = customer_features.index[customer_features['CustomerID'] == customer_id].tolist()
    if not customer_idx:
        return []
    customer_idx = customer_idx[0]
    
    # Get similarity scores for the customer
    similar_scores = similarity_matrix[customer_idx]
    
    # Get indices of top similar customers (excluding self)
    similar_indices = similar_scores.argsort()[-(n_recommendations + 1):-1][::-1]
    
    # Retrieve customer IDs and similarity scores
    lookalikes = []
    for idx in similar_indices:
        similar_customer_id = customer_features.iloc[idx]['CustomerID']
        score = similar_scores[idx]
        lookalikes.append({
            'similar_customer_id': similar_customer_id,
            'similarity_score': round(float(score), 4)  # Rounded for readability
        })
    
    return lookalikes

def train_model(customer_features):
    """Train a Random Forest model to predict customer similarity."""
    # Prepare features and labels
    X = customer_features.drop(columns=['CustomerID'])
    y = (X['TotalValue_sum'] > X['TotalValue_sum'].mean()).astype(int)  # Example binary target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return model

def main():
    # Load data
    customers_df, products_df, transactions_df = load_data()
    
    # Create customer features
    customer_features = create_advanced_features(customers_df, transactions_df, products_df)
    
    # Select only numerical features for similarity calculation
    numerical_features = customer_features.select_dtypes(include=[np.number])
    
    # Check if 'CustomerID' exists before dropping
    if 'CustomerID' in numerical_features.columns:
        numerical_features = numerical_features.drop(columns=['CustomerID'])
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(numerical_features)
    
    # Calculate similarity scores
    similarity_matrix = calculate_similarity_scores(features_scaled)
    
    # Train the model
    model = train_model(customer_features)
    
    # Generate lookalikes for first 20 customers
    results = []
    first_20_customers = customers_df['CustomerID'].iloc[:20].tolist()
    for cust_id in first_20_customers:
        lookalikes = find_lookalikes(cust_id, customer_features, similarity_matrix, n_recommendations=3)
        results.append({
            'customer_id': cust_id,
            'lookalikes': lookalikes
        })
    
    # Convert results to a DataFrame
    lookalike_df = pd.DataFrame(results)
    
    # Expand lookalikes into separate rows for better readability (optional)
    lookalike_expanded = lookalike_df.explode('lookalikes').reset_index(drop=True)
    lookalike_expanded[['similar_customer_id', 'similarity_score']] = lookalike_expanded['lookalikes'].apply(pd.Series)
    lookalike_expanded = lookalike_expanded.drop(columns=['lookalikes'])
    
    # Save to CSV
    lookalike_expanded.to_csv('Mrunal_Deshmukh_Lookalike.csv', index=False)
    
    print("Lookalike recommendations saved to 'Mrunal_Deshmukh_Lookalike.csv'.")

if __name__ == "__main__":
    main()