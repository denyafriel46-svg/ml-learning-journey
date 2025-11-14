# Project 8: Decision Tree - Breast Cancer Prediction

## Description
Model machine learning menggunakan Decision Tree untuk memprediksi apakah tumor kanker payudara bersifat malignant (ganas) atau benign (jinak).

## Dataset
- Breast Cancer Wisconsin dataset dari sklearn
- Total samples: 569 pasien
- Features: 30 karakteristik (radius, texture, perimeter, dll)
- Classes: 2 kategori (Malignant, Benign)

## Model
- Algorithm: Decision Tree Classifier
- Max Depth: 3
- Train-Test Split: 80-20

## Results
- Train Accuracy: 97.8%
- Test Accuracy: 94.7%
- Precision: 95%
- Recall: 94%

## Features
Model mengidentifikasi fitur paling penting:
- Mean concave points: 75.23% (paling penting!)
- Worst radius: 5.69%
- Worst perimeter: 5.60%

## How to Run
```bash
python decision_trees.py
```

## Output
- Console: Predictions & Metrics
- Image: `decision_tree_output.png` (visualisasi)

## Key Learnings
- Train vs Test Accuracy comparison
- Feature importance analysis
- Confusion Matrix untuk error analysis
- Prediction confidence levels