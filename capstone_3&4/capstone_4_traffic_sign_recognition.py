# Capstone Project 4: Smart Traffic Sign Recognition System
# Stage 1: Learning Basic Neural Networks (Perceptron)

# Scenario:
# City is building a Smart Traffic Monitoring System
# System must recognize traffic signs: Stop, Speed Limit, Pedestrian Crossing
# Stage 1: Implement Perceptron to predict "STOP" vs "NOT STOP"

# Dataset: https://github.com/himanshusar123/Datasets (DatasetCapstoneProject4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================
# Generate Synthetic Traffic Sign Dataset
# ============================================
print("=" * 60)
print("SMART TRAFFIC SIGN RECOGNITION SYSTEM")
print("Stage 1: Perceptron-Based Neural Network")
print("=" * 60)

print("\nGenerating synthetic traffic sign dataset...")
np.random.seed(42)

n_samples = 500

# Features extracted from traffic signs
data = {
    'red_intensity': np.random.randint(0, 255, n_samples),
    'shape_circularity': np.random.uniform(0, 1, n_samples),
    'size_pixels': np.random.randint(50, 500, n_samples),
    'edge_sharpness': np.random.uniform(0, 1, n_samples),
    'color_saturation': np.random.uniform(0, 1, n_samples),
    'brightness': np.random.randint(0, 255, n_samples),
}

df = pd.DataFrame(data)

# Generate target: STOP sign (1) or NOT STOP (0)
# STOP signs typically have: high red intensity, circular shape, high edge sharpness
stop_prob = (
    (df['red_intensity'] / 255) * 0.4 +
    df['shape_circularity'] * 0.3 +
    df['edge_sharpness'] * 0.2 +
    np.random.random(n_samples) * 0.1
)

df['is_stop_sign'] = (stop_prob > 0.5).astype(int)

print("✓ Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"\nFirst 5 samples:")
print(df.head())

print(f"\nTarget distribution:")
print(df['is_stop_sign'].value_counts())
print(f"\nSTOP signs: {df['is_stop_sign'].sum()} ({df['is_stop_sign'].mean()*100:.1f}%)")
print(f"NOT STOP signs: {(1-df['is_stop_sign']).sum()} ({(1-df['is_stop_sign']).mean()*100:.1f}%)")

# ============================================
# Perceptron Implementation
# ============================================
print("\n" + "=" * 60)
print("PERCEPTRON NEURAL NETWORK")
print("=" * 60)

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []
    
    def activation(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        """Train the perceptron"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        print(f"\nTraining Perceptron...")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_errors = 0
            
            for idx, x_i in enumerate(X):
                # Calculate output
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                
                # Calculate error
                error = y[idx] - y_pred
                
                # Update weights and bias
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
                
                epoch_errors += int(error != 0)
            
            self.errors.append(epoch_errors)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Errors = {epoch_errors}")
        
        print(f"\n✓ Training complete!")
    
    def predict(self, X):
        """Make predictions"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# ============================================
# Prepare Data
# ============================================
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Separate features and target
X = df.drop('is_stop_sign', axis=1).values
y = df['is_stop_sign'].values

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Feature scaling complete")

# ============================================
# Train Perceptron
# ============================================
print("\n" + "=" * 60)
print("TRAINING PERCEPTRON MODEL")
print("=" * 60)

perceptron = Perceptron(learning_rate=0.1, epochs=50)
perceptron.fit(X_train_scaled, y_train)

# Make predictions
y_pred = perceptron.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['NOT STOP', 'STOP']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================
# Visualization
# ============================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target Distribution
df['is_stop_sign'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['blue', 'red'])
axes[0, 0].set_title('Traffic Sign Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Sign Type')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticklabels(['NOT STOP', 'STOP'], rotation=0)

# 2. Training Errors
axes[0, 1].plot(perceptron.errors, marker='o', color='orange')
axes[0, 1].set_title('Perceptron Training Errors', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Number of Errors')
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['NOT STOP', 'STOP'],
            yticklabels=['NOT STOP', 'STOP'])
axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 4. Feature Importance (Weights)
feature_names = df.drop('is_stop_sign', axis=1).columns
weights_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': perceptron.weights
}).sort_values('Weight', ascending=True)

axes[1, 1].barh(weights_df['Feature'], weights_df['Weight'], color='green')
axes[1, 1].set_title('Perceptron Feature Weights', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Weight Value')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('capstone_3&4/traffic_sign_perceptron_results.png', dpi=300)
print("✓ Visualization saved: traffic_sign_perceptron_results.png")
plt.show()

# ============================================
# Feature Analysis
# ============================================
print("\n" + "=" * 60)
print("FEATURE ANALYSIS")
print("=" * 60)

print("\nPerceptron Weights:")
for feature, weight in zip(feature_names, perceptron.weights):
    print(f"  {feature:20s}: {weight:8.4f}")

print(f"\nBias: {perceptron.bias:.4f}")

print("\n💡 Feature Importance Interpretation:")
print("  • Positive weights: Increase likelihood of STOP sign")
print("  • Negative weights: Decrease likelihood of STOP sign")
print("  • Larger absolute values: More important features")

# ============================================
# Business Insights
# ============================================
print("\n" + "=" * 60)
print("SYSTEM INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("\n🚦 MODEL PERFORMANCE:")
print(f"  • Accuracy: {accuracy*100:.2f}%")
print(f"  • Successfully classifies traffic signs")

print("\n🎯 KEY FINDINGS:")
most_important = weights_df.iloc[-1]
print(f"  • Most important feature: {most_important['Feature']}")
print(f"  • Weight value: {most_important['Weight']:.4f}")

print("\n🚗 REAL-WORLD APPLICATIONS:")
print("  1. Autonomous Vehicles: Real-time sign detection")
print("  2. Traffic Monitoring: Automated sign recognition")
print("  3. Driver Assistance: Alert systems for traffic signs")
print("  4. Smart Cities: Traffic flow optimization")

print("\n📈 NEXT STEPS (Stage 2):")
print("  • Implement CNN for image-based recognition")
print("  • Add more sign categories (Speed Limit, Pedestrian)")
print("  • Handle real camera images with noise")
print("  • Deploy on edge devices for real-time processing")

print("\n💰 EXPECTED IMPACT:")
print("  • Improved road safety through automated detection")
print("  • Reduced human error in sign recognition")
print("  • Support for autonomous vehicle navigation")
print("  • Enhanced traffic monitoring efficiency")

print("\n" + "=" * 60)
print("✓ STAGE 1 COMPLETE!")
print("=" * 60)
print("\nPerceptron successfully learned to classify traffic signs!")
print("Ready for Stage 2: CNN-based image recognition")
