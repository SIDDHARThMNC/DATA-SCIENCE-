import numpy as np

# Daily sales data
sales_day1 = np.array([120, 85, 60])  # Product A, B, C
sales_day2 = np.array([150, 90, 75])

print("=== Retail Sales Analysis ===\n")

# 1. Find the total sales per product across the two days
total_sales = sales_day1 + sales_day2
print("1. Total sales per product (A, B, C):")
print(f"   Product A: {total_sales[0]}")
print(f"   Product B: {total_sales[1]}")
print(f"   Product C: {total_sales[2]}")
print(f"   Array: {total_sales}\n")

# 2. Compute the percentage growth from day 1 to day 2
percentage_growth = ((sales_day2 - sales_day1) / sales_day1) * 100
print("2. Percentage growth from day 1 to day 2:")
print(f"   Product A: {percentage_growth[0]:.2f}%")
print(f"   Product B: {percentage_growth[1]:.2f}%")
print(f"   Product C: {percentage_growth[2]:.2f}%")
print(f"   Array: {percentage_growth}\n")

# 3. Which product had the highest growth?
highest_growth_index = np.argmax(percentage_growth)
products = ['A', 'B', 'C']
print("3. Product with highest growth:")
print(f"   Product {products[highest_growth_index]} with {percentage_growth[highest_growth_index]:.2f}% growth")
