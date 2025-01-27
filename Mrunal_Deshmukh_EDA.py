import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    """Load all required datasets"""
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert dates
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    return customers_df, products_df, transactions_df

def basic_eda(df, name):
    """Perform basic EDA on a dataframe"""
    print(f"\n{name} Dataset Analysis:")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())

def sales_analysis(transactions_df, products_df):
    """Analyze sales patterns"""
    # Monthly sales trend
    monthly_sales = transactions_df.groupby(
        transactions_df['TransactionDate'].dt.to_period('M')
    )['TotalValue'].sum()
    
    # Plot monthly sales
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.savefig('monthly_sales.png')
    plt.close()

def customer_analysis(customers_df, transactions_df):
    """Analyze customer behavior"""
    # Customer purchase frequency
    customer_purchases = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    customer_purchases.columns = ['CustomerID', 'Purchase_Count', 'Total_Spend', 'Total_Quantity']
    return customer_purchases

def main():
    # Load data
    customers_df, products_df, transactions_df = load_data()
    
    # Perform basic EDA
    basic_eda(customers_df, "Customers")
    basic_eda(products_df, "Products")
    basic_eda(transactions_df, "Transactions")
    
    # Perform detailed analysis
    sales_analysis(transactions_df, products_df)
    customer_metrics = customer_analysis(customers_df, transactions_df)
    
    

if __name__ == "__main__":
    main() 