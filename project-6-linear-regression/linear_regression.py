import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("PROJECT 6: LINEAR REGRESSION - PREDICT HOUSE PRICE")
print("="*60)

# Step 1: Buat dataset
# Data: ukuran rumah (m2) dan harganya (juta rupiah)
data = {
    'Ukuran_m2': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'Harga_Juta': [300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

print("\n" + "="*60)

# Step 2: Prepare data
X = df[['Ukuran_m2']].values  # Feature (input) - yang dipakai untuk predict
y = df['Harga_Juta'].values   # Target (output) - yang mau di-predict

print(f"\nX (Ukuran): \n{X.flatten()}")
print(f"y (Harga): \n{y}")

print("\n" + "="*60)

# Step 3: Split data - train dan test
# 80% untuk training, 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} samples")

print("\n" + "="*60)

# Step 4: Create dan train model
model = LinearRegression()
model.fit(X_train, y_train)  # Train model dengan training data

print("\n✓ Model trained!")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print("Artinya: Harga = {:.2f} * Ukuran + {:.2f}".format(model.coef_[0], model.intercept_))

print("\n" + "="*60)

# Step 5: Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nPredictions (Train data):")
for i in range(len(X_train)):
    print(f"Ukuran: {X_train[i][0]} m2 → Actual: {y_train[i]} juta, Predicted: {y_pred_train[i]:.2f} juta")

print("\nPredictions (Test data):")
for i in range(len(X_test)):
    print(f"Ukuran: {X_test[i][0]} m2 → Actual: {y_test[i]} juta, Predicted: {y_pred_test[i]:.2f} juta")

print("\n" + "="*60)

# Step 6: Evaluate model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\nModel Evaluation:")
print(f"Train MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")
print(f"Train R² Score: {r2_train:.4f}")
print(f"Test R² Score: {r2_test:.4f}")
print("\nR² Score explanation:")
print("- R² = 1.0 → Model perfect")
print("- R² = 0.5 → Model 50% akurat")
print("- R² = 0.0 → Model jelek")

print("\n" + "="*60)

# Step 7: Predict new data
new_ukuran = np.array([[75], [125]])  # Rumah dengan ukuran 75 m2 dan 125 m2
new_predictions = model.predict(new_ukuran)

print("\nPredict new data:")
for i in range(len(new_ukuran)):
    print(f"Rumah {i+1} ukuran {new_ukuran[i][0]} m2 → Predicted price: {new_predictions[i]:.2f} juta")

print("\n" + "="*60)

# Step 8: Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Scatter + regression line (train data)
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Train Data', s=100)
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Regression Line')
plt.scatter(new_ukuran, new_predictions, color='green', marker='*', s=300, label='New Predictions')
plt.xlabel('Ukuran (m2)')
plt.ylabel('Harga (juta)')
plt.title('Linear Regression - House Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (test data)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='purple', s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (juta)')
plt.ylabel('Predicted Price (juta)')
plt.title('Actual vs Predicted (Test Data)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_output.png', dpi=100)
print("\n✓ Plot saved as 'linear_regression_output.png'")
# plt.show()

print("\n" + "="*60)
print("\nSELESAI PROJECT 6!")
print("="*60)