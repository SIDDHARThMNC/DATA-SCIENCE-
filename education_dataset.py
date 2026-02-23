import numpy as np

# Student scores in three subjects [Math, English, Science]
scores = np.array([[85, 78, 92],
                   [88, 74, 90],
                   [70, 65, 80],
                   [95, 89, 96]])

print("=== Education Dataset (Indexing & Slicing) ===\n")
print("Student scores [Math, English, Science]:")
print("         Math  English  Science")
for i, student_scores in enumerate(scores):
    print(f"Student {i}: {student_scores}")
print()

# 1. Extract scores of the top two students
top_two_students = scores[:2]
print("1. Scores of the top two students (first 2 rows):")
print("         Math  English  Science")
for i, student_scores in enumerate(top_two_students):
    print(f"Student {i}: {student_scores}")
print()

# 2. Get all Math scores (first column)
math_scores = scores[:, 0]
print("2. All Math scores (first column):")
print(f"   {math_scores}")
print()

# 3. Find students who scored > 90 in Science (third column)
science_scores = scores[:, 2]
high_science_mask = science_scores > 90
high_science_students = np.where(high_science_mask)[0]

print("3. Students who scored > 90 in Science (third column):")
print(f"   Science scores: {science_scores}")
print(f"   Boolean mask (> 90): {high_science_mask}")
print(f"   Student indices: {high_science_students}")
print()
print("   Student details:")
for idx in high_science_students:
    print(f"      Student {idx}: Math={scores[idx, 0]}, English={scores[idx, 1]}, Science={scores[idx, 2]}")
