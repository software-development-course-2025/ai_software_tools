# -*- coding: utf-8 -*-
#
# Task 1: Classical ML with Scikit-learn
# Objective: Train a Decision Tree Classifier on the Iris Dataset and evaluate performance.
# Framework: Scikit-learn (Classical ML)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
import numpy as np

# Suppress warnings for clearer output 
warnings.filterwarnings('ignore')

print("--- Task 1: Decision Tree Classification (Iris Dataset) ---")

# 1. Data Loading and Preprocessing 
try:
    # Load the Iris dataset 
    iris = load_iris(as_frame=True)
    X = iris.data
    y_raw = iris.target
    target_names = iris.target_names

    print(f"\nFeatures: {list(X.columns)}")
    print(f"Classes: {target_names}")
    
    # 2. Data Splitting 
    # 70% for training, 30% for testing. Stratify ensures equal representation of classes.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )

    print(f"\nData Split:")
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")

except Exception as e:
    print(f"Error loading Iris Dataset: {e}")
    exit()

# 3. Model Training: Decision Tree Classifier 
print("\n3. Training the Decision Tree Classifier...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("   Training finished.")

# 4. Model Evaluation 
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# Use 'macro' average to treat all classes equally 
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')

print("\n--- Evaluation Results (30% Test Set) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")

# Detailed Classification Report 
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nTask 1 completed successfully.")
