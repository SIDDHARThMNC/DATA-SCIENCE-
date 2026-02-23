import numpy as np

# Patient blood pressure readings [systolic, diastolic]
bp = np.array([[120, 80],
               [135, 85],
               [140, 90],
               [110, 70],
               [125, 75]])

print("=== Healthcare Dataset (Indexing) ===\n")
print("Original blood pressure readings [systolic, diastolic]:")
print(bp)
print()

# 1. Extract the systolic values (first column)
systolic = bp[:, 0]
print("1. Systolic values (first column):")
print(f"   {systolic}")
print()

# 2. Find patients with systolic > 130 using boolean indexing
high_systolic_mask = systolic > 130
high_systolic_patients = np.where(high_systolic_mask)[0]
print("2. Patients with systolic > 130:")
print(f"   Boolean mask: {high_systolic_mask}")
print(f"   Patient indices: {high_systolic_patients}")
print(f"   Patient readings:")
for idx in high_systolic_patients:
    print(f"      Patient {idx}: {bp[idx]}")
print()

# 3. Replace all diastolic values < 80 with 80 (minimum threshold)
bp_corrected = bp.copy()  # Create a copy to preserve original
diastolic_mask = bp_corrected[:, 1] < 80
bp_corrected[diastolic_mask, 1] = 80

print("3. Replace diastolic values < 80 with 80:")
print(f"   Diastolic values < 80 mask: {diastolic_mask}")
print(f"   Corrected blood pressure readings:")
print(bp_corrected)
print()
print("   Changes made:")
for idx in np.where(diastolic_mask)[0]:
    print(f"      Patient {idx}: {bp[idx]} → {bp_corrected[idx]}")
