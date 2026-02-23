"""
E-Commerce Customer & Sales Data Wrangling
Assignment: Complete data preparation pipeline for ML/DL models
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("E-COMMERCE DATA WRANGLING PROJECT")
print("="*80)

# ============================================================================
# TASK 1: DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("TASK 1: DATA LOADING")
print("="*80)

# Load datasets
customers_df = pd.read_csv('Customers1 - Sheet1.csv')
sales_df = pd.read_csv('Sales1 - Sheet1.csv')
support_df = pd.read_csv('support1 - Sheet1.csv')

print("\n--- Customers Dataset ---")
print(f"Shape: {customers_df.shape}")
print(f"Columns: {list(customers_df.columns)}")
print(f"Missing Values:\n{customers_df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{customers_df.head()}")

print("\n--- Sales Dataset ---")
print(f"Shape: {sales_df.shape}")
print(f"Columns: {list(sales_df.columns)}")
print(f"Missing Values:\n{sales_df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{sales_df.head()}")

print("\n--- Support Dataset ---")
print(f"Shape: {support_df.shape}")
print(f"Columns: {list(support_df.columns)}")
print(f"Missing Values:\n{support_df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{support_df.head()}")

# ============================================================================
# TASK 2: ARRAY OPERATIONS & BROADCASTING (NumPy Integration)
# ============================================================================
print("\n" + "="*80)
print("TASK 2: ARRAY OPERATIONS & BROADCASTING")
print("="*80)

# Create NumPy array of prices and apply 10% discount
prices_array = sales_df['Price'].values
discount_rate = 0.10
discounted_prices = prices_array * (1 - discount_rate)

print(f"\nOriginal Prices (NumPy Array):\n{prices_array}")
print(f"\nDiscounted Prices (10% off using Broadcasting):\n{discounted_prices}")

# Compute total revenue per order (Quantity × Price)
sales_df['Revenue'] = sales_df['Quantity'] * sales_df['Price']
print(f"\nRevenue per Order:\n{sales_df[['OrderID', 'Quantity', 'Price', 'Revenue']]}")

# ============================================================================
# TASK 3: INDEXING & SLICING
# ============================================================================
print("\n" + "="*80)
print("TASK 3: INDEXING & SLICING")
print("="*80)

# Convert OrderDate to datetime
sales_df['OrderDate'] = pd.to_datetime(sales_df['OrderDate'])

# Extract orders from January 2025 (Note: sample data is from 2023, showing logic)
january_2025_orders = sales_df[
    (sales_df['OrderDate'].dt.month == 1) & 
    (sales_df['OrderDate'].dt.year == 2025)
]
print(f"\nOrders placed in January 2025:")
print(january_2025_orders if len(january_2025_orders) > 0 else "No orders in January 2025")

# Slice first 10 rows
print(f"\nFirst 10 rows of Sales Dataset:")
print(sales_df.head(10))

# ============================================================================
# TASK 4: FILTERING
# ============================================================================
print("\n" + "="*80)
print("TASK 4: FILTERING")
print("="*80)

# Filter customers from North region
north_customers = customers_df[customers_df['Region'] == 'North']
print(f"\nCustomers from North Region:")
print(north_customers)

# Identify orders with revenue > ₹10,000
high_revenue_orders = sales_df[sales_df['Revenue'] > 10000]
print(f"\nOrders with Revenue > ₹10,000:")
print(high_revenue_orders[['OrderID', 'CustomerID', 'Product', 'Revenue']])

# ============================================================================
# TASK 5: SORTING
# ============================================================================
print("\n" + "="*80)
print("TASK 5: SORTING")
print("="*80)

# Convert SignupDate to datetime
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])

# Sort customers by Signup Date (oldest to newest)
customers_sorted = customers_df.sort_values('SignupDate', ascending=True)
print(f"\nCustomers sorted by Signup Date (Oldest to Newest):")
print(customers_sorted[['CustomerID', 'Name', 'SignupDate']])

# Sort sales by Revenue (descending)
sales_sorted = sales_df.sort_values('Revenue', ascending=False)
print(f"\nSales sorted by Revenue (Highest to Lowest):")
print(sales_sorted[['OrderID', 'CustomerID', 'Product', 'Revenue']])

# ============================================================================
# TASK 6: GROUPING
# ============================================================================
print("\n" + "="*80)
print("TASK 6: GROUPING")
print("="*80)

# Merge sales with customers to get Region info
sales_with_region = sales_df.merge(customers_df[['CustomerID', 'Region']], on='CustomerID', how='left')

# Group sales by Region and calculate average revenue
avg_revenue_by_region = sales_with_region.groupby('Region')['Revenue'].mean()
print(f"\nAverage Revenue by Region:")
print(avg_revenue_by_region)

# Group support tickets by Issue Type and find average resolution time
avg_resolution_by_issue = support_df.groupby('IssueType')['ResolutionTime'].mean()
print(f"\nAverage Resolution Time by Issue Type (hours):")
print(avg_resolution_by_issue)

# ============================================================================
# TASK 7: DATA WRANGLING
# ============================================================================
print("\n" + "="*80)
print("TASK 7: DATA WRANGLING")
print("="*80)

# Handle missing values - fill missing ages with median age
median_age = customers_df['Age'].median()
customers_df['Age'] = customers_df['Age'].fillna(median_age)
print(f"\nMissing ages filled with median age: {median_age}")
print(f"Updated Customers Dataset:\n{customers_df}")

# Rename columns for clarity (already using CustomerID, but showing the concept)
support_df.rename(columns={'TicketID': 'Ticket_ID', 'IssueType': 'Issue_Type', 
                           'ResolutionTime': 'Resolution_Time'}, inplace=True)
print(f"\nSupport Dataset with renamed columns:\n{support_df.head()}")

# Merge all datasets on CustomerID
# First merge customers with sales
merged_df = customers_df.merge(sales_df, on='CustomerID', how='left')

# Then merge with support
merged_df = merged_df.merge(support_df, on='CustomerID', how='left')

print(f"\nMerged Dataset Shape: {merged_df.shape}")
print(f"\nMerged Dataset Preview:\n{merged_df.head()}")

# Create calculated fields
# Customer Lifetime Value (CLV) = total revenue per customer
clv = sales_df.groupby('CustomerID')['Revenue'].sum().reset_index()
clv.rename(columns={'Revenue': 'CLV'}, inplace=True)

# Average Resolution Time per Customer
avg_resolution_per_customer = support_df.groupby('CustomerID')['Resolution_Time'].mean().reset_index()
avg_resolution_per_customer.rename(columns={'Resolution_Time': 'Avg_Resolution_Time'}, inplace=True)

# Create final cleaned dataset
final_df = customers_df.merge(clv, on='CustomerID', how='left')
final_df = final_df.merge(avg_resolution_per_customer, on='CustomerID', how='left')

# Fill NaN values in CLV and Avg_Resolution_Time with 0
final_df['CLV'] = final_df['CLV'].fillna(0)
final_df['Avg_Resolution_Time'] = final_df['Avg_Resolution_Time'].fillna(0)

print(f"\nFinal Cleaned Dataset with Calculated Fields:")
print(final_df)

# Export to CSV
final_df.to_csv('Cleaned_Data.csv', index=False)
print(f"\n✓ Cleaned dataset exported to 'Cleaned_Data.csv'")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR ML/DL MODEL")
print("="*80)

print(f"\nTotal Customers: {len(customers_df)}")
print(f"Total Orders: {len(sales_df)}")
print(f"Total Support Tickets: {len(support_df)}")
print(f"Total Revenue: ₹{sales_df['Revenue'].sum():,.2f}")
print(f"Average Order Value: ₹{sales_df['Revenue'].mean():,.2f}")
print(f"Average Customer Lifetime Value: ₹{final_df['CLV'].mean():,.2f}")
print(f"Average Resolution Time: {support_df['Resolution_Time'].mean():.2f} hours")

print("\n" + "="*80)
print("DATA WRANGLING COMPLETED SUCCESSFULLY!")
print("Dataset is ready for ML/DL model training")
print("="*80)
