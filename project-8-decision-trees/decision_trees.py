import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

print("PROJECT 8: breast_cancer - PREDICT CANCER  TYPE")
print("="*60)

# Step 1: Load dataset (breast cancer - dataset terkenal)
cancer = load_breast_cancer()
X = cancer.data  # Features: sepal length, sepal width, petal length, petal width
y = cancer.target  # Target: 0=malignant (ganas), 1=benign (jinak)

print("Dataset: breast cancer")
print(f"Total samples: {len(X)}")
print(f"Features: {cancer.feature_names}")
print(f"Classes: {cancer.target_names}")

# Create DataFrame untuk lebih mudah dibaca
df = pd.DataFrame(X, columns=cancer.feature_names)
df['diagnosis'] = cancer.target_names[y]
print("\nDataset Preview:")
print(df.head(10))

print("\n" + "="*60)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} samples")

print("\n" + "="*60)

# Step 3: Create dan train Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("\n✓ Decision Tree model trained!")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

print("\n" + "="*60)

# Step 4: Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nSample Predictions (Test data):")
for i in range(min(10, len(X_test))):
    actual = cancer.target_names[y_test[i]]
    predicted = cancer.target_names[y_pred_test[i]]
    match = "✓" if actual == predicted else "✗"
    print(f"{match} Actual: {actual:12} → Predicted: {predicted:12}")

print("\n" + "="*60)

# Step 5: Evaluate model
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("\nModel Evaluation:")
print(f"Train Accuracy: {accuracy_train:.4f} ({accuracy_train*100:.1f}%)")
print(f"Test Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.1f}%)")

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test, target_names=cancer.target_names))

print("\n" + "="*60)

# Step 6: Feature Importance
feature_importance = model.feature_importances_
print("\nFeature Importance:")
for i, importance in enumerate(feature_importance):
    print(f"{cancer.feature_names[i]:25} : {importance:.4f}")

print("\nArtinya: Feature mana yang paling penting untuk classify")

print("\n" + "="*60)

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

print("\n" + "="*60)

# Step 8: Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Decision Tree Structure
ax1 = axes[0, 0]
plot_tree(model, feature_names=cancer.feature_names, class_names=cancer.target_names,
          filled=True, ax=ax1, fontsize=8)
ax1.set_title('Decision Tree Structure', fontsize=12, fontweight='bold')

# Plot 2: Feature Importance
ax2 = axes[0, 1]
colors = ['blue', 'green', 'orange', 'red']
ax2.barh(cancer.feature_names, feature_importance, color=colors, alpha=0.7)
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Confusion Matrix Heatmap
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=cancer.target_names, yticklabels=cancer.target_names, ax=ax3)
ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# Plot 4: Accuracy Comparison
ax4 = axes[1, 1]
accuracies = [accuracy_train, accuracy_test]
labels = ['Train', 'Test']
colors_acc = ['green', 'blue']
bars = ax4.bar(labels, accuracies, color=colors_acc, alpha=0.7)
ax4.set_ylabel('Accuracy')
ax4.set_title('Train vs Test Accuracy', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1.1])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('decision_tree_output.png', dpi=100)
print("\n✓ Plots saved as 'decision_tree_output.png'")
# plt.show()

print("\n" + "="*60)
print("\nSELESAI PROJECT 8!")
print("="*60)

# Predict pasien baru
print("\n" + "="*60)
print("PREDICT PASIEN BARU:")
print("="*60)

# Contoh 1: Pasien dengan karakteristik tertentu
pasien_1 = np.array([[17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.0950, 0.9053, 8.589, 153.40, 0.0064, 0.0049, 0.0037, 0.0030, 0.0064, 0.0025, 0.4992, 0.2062, 0.4505, 0.2430, 0.3613, 0.0876, 0.6638, 0.2564,0.3430,0.5030]])

prediksi_1 = model.predict(pasien_1)
print(f"\nPasien 1: {cancer.target_names[prediksi_1[0]]}")
print(f"Confidence: {model.predict_proba(pasien_1)[0]}")

# Contoh 2: Pasien berbeda
pasien_2 = np.array([[13.54, 14.36, 87.46, 566.3, 0.1091, 0.0613, 0.0749, 0.0490, 0.0837, 0.0467, 0.1090, 0.1270, 0.7798, 86.20, 0.0060, 0.0039, 0.0059, 0.0037, 0.0086, 0.0070, 0.2565, 0.1236, 0.5387, 0.6130, 0.2614, 0.0860, 0.5090, 0.1765,0.5030,0.5094]])

prediksi_2 = model.predict(pasien_2)
print(f"\nPasien 2: {cancer.target_names[prediksi_2[0]]}")
print(f"Confidence: {model.predict_proba(pasien_2)[0]}")