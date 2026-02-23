# Scenario: Farm Crop Yield Analysis
# A farm tracks the yield (in kilograms) of three different crops — Wheat, Rice, and Corn — across four seasons (Spring, Summer, Autumn, Winter).
# The farm manager wants to analyze the data to make better decisions for next year.

import numpy as np

# Create a 2D array representing crop yields for 3 crops across 4 seasons
# Rows = crops (Wheat, Rice, Corn)
# Columns = seasons (Spring, Summer, Autumn, Winter)
crop_yields = np.array([
    [2800, 3200, 2900, 2600],  # Wheat
    [3100, 3400, 3000, 2800],  # Rice
    [2700, 3300, 3100, 2500]   # Corn
])

print("Original Crop Yields (kg):")
print("Crops: Wheat, Rice, Corn")
print("Seasons: Spring, Summer, Autumn, Winter")
print(crop_yields)
print()

# Increase all yields by 10% to account for improved irrigation
improved_yields = crop_yields * 1.10
print("Improved Yields (10% increase):")
print(improved_yields)
print()

# Extract only Wheat's yields (the first crop)
wheat_yields = crop_yields[0]
print("Wheat Yields Across Seasons:")
print(wheat_yields)
print()

# Find the average yield per crop across all seasons
average_per_crop = np.mean(crop_yields, axis=1)
print("Average Yield Per Crop (across all seasons):")
print(f"Wheat: {average_per_crop[0]:.2f} kg")
print(f"Rice: {average_per_crop[1]:.2f} kg")
print(f"Corn: {average_per_crop[2]:.2f} kg")
print()

# Identify yields greater than 3000 kg
high_yields = crop_yields > 3000
print("Yields Greater Than 3000 kg (Boolean mask):")
print(high_yields)
print()

print("Specific High-Performing Yields:")
print(crop_yields[high_yields])
