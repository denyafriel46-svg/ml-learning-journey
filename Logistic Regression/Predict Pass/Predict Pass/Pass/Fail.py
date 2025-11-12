import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

print("PROJECT 7: LOGISTIC REGRESSION - PREDICT PASS/FAIL")
print("="*60)

# Step 1: Buat dataset
# Data: jam belajar dan hasil (0=Gagal, 1=Lulus)
data = {
    'Jam_Belajar': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Hasil': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0=Gagal, 1=Lulus
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\nLegend: 0=Gagal, 1=Lulus")

print("\n" + "="*60)

# Step 2: Prepare data
X = df[['Jam_Belajar']].values
y = df['Hasil'].values

print(f"\nX (Jam Belajar): {X.flatten()}")
print(f"y (Hasil): {y}")

print("\n" + "="*60)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} samples")

print("\n" + "="*60)

# Step 4: Create dan train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("\n✓ Model trained!")
print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

print("\n" + "="*60)

# Step 5: Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nPredictions (Train data):")
for i in range(len(X_train)):
    actual = "Lulus" if y_train[i] == 1 else "Gagal"
    predicted = "Lulus" if y_pred_train[i] == 1 else "Gagal"
    print(f"Jam {X_train[i][0]} → Actual: {actual}, Predicted: {predicted}")

print("\nPredictions (Test data):")
for i in range(len(X_test)):
    actual = "Lulus" if y_test[i] == 1 else "Gagal"
    predicted = "Lulus" if y_pred_test[i] == 1 else "Gagal"
    print(f"Jam {X_test[i][0]} → Actual: {actual}, Predicted: {predicted}")

print("\n" + "="*60)

# Step 6: Evaluate model
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, zero_division=0)
recall = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

print("\nModel Evaluation:")
print(f"Train Accuracy: {accuracy_train:.4f} ({accuracy_train*100:.1f}%)")
print(f"Test Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.1f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nMetric explanation:")
print("- Accuracy: Berapa % prediksi benar")
print("- Precision: Dari yang di-predict Lulus, berapa % benar Lulus")
print("- Recall: Dari yang seharusnya Lulus, berapa % yang ter-detect")
print("- F1: Balance antara precision & recall")

print("\n" + "="*60)

# Step 7: Predict probability
print("\nPredict probability (seberapa yakin):")
new_jam = np.array([[3], [7]])
probabilities = model.predict_proba(new_jam)

for i in range(len(new_jam)):
    jam = new_jam[i][0]
    prob_gagal = probabilities[i][0] * 100
    prob_lulus = probabilities[i][1] * 100
    print(f"Jam {jam} belajar → Gagal: {prob_gagal:.1f}%, Lulus: {prob_lulus:.1f}%")

print("\n" + "="*60)

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)
print("- Top-left: True Negatives (Correctly predicted Gagal)")
print("- Top-right: False Positives (Predicted Lulus, tapi Gagal)")
print("- Bottom-left: False Negatives (Predicted Gagal, tapi Lulus)")
print("- Bottom-right: True Positives (Correctly predicted Lulus)")

print("\n" + "="*60)

# Step 9: Visualization
plt.figure(figsize=(14, 5))

# Plot 1: Logistic Regression Curve
plt.subplot(1, 3, 1)
X_range = np.linspace(0, 11, 300).reshape(-1, 1)
y_proba = model.predict_proba(X_range)[:, 1]

plt.scatter(X[y == 0], y[y == 0], color='red', label='Gagal', s=100)
plt.scatter(X[y == 1], y[y == 1], color='green', label='Lulus', s=100)
plt.plot(X_range, y_proba, color='blue', linewidth=2, label='Decision Boundary')
plt.xlabel('Jam Belajar')
plt.ylabel('Probability Lulus')
plt.title('Logistic Regression - Pass/Fail Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Confusion Matrix Heatmap
plt.subplot(1, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 3: Metrics Comparison
plt.subplot(1, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy_test, precision, recall, f1]
colors = ['blue', 'green', 'orange', 'red']
plt.bar(metrics, values, color=colors, alpha=0.7)
plt.ylabel('Score')
plt.title('Model Metrics')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('logistic_regression_output.png', dpi=100)
print("\n✓ Plot saved as 'logistic_regression_output.png'")
# plt.show()

print("\n" + "="*60)
print("\nSELESAI PROJECT 7!")
print("="*60)