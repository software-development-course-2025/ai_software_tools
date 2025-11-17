# -*- coding: utf-8 -*-
#
# Task 2: Deep Learning with TensorFlow/Keras
# Objective: Build and train a CNN to classify handwritten digits (MNIST)
# Framework: TensorFlow / Keras (Deep Learning)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Ensure TensorFlow runs only once for logging 
print("--- Task 2: CNN Image Classification (MNIST Dataset) ---")

# 1. Load and Preprocess Data 
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape: Add channel dimension (28x28 -> 28x28x1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Normalize: Scale pixel values from 0-255 to 0-1 
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # One-Hot Encoding: Convert labels to binary vectors 
    num_classes = 10
    y_train_encoded = to_categorical(y_train, num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)

    print(f"\nTraining data shape: {x_train.shape}")

except Exception as e:
    print(f"Error loading or preprocessing MNIST data: {e}")
    exit()


# 2. Define the CNN Model Architecture 
model = Sequential([
    # Input layer and first Convolution 
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolution
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Dropout(0.25),
    
    # Fully Connected Layers 
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    # Output Layer 
    Dense(num_classes, activation='softmax')
])

# 3. Compile the Model 
print("\n3. Compiling Model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Loss function for multi-class classification 
    metrics=['accuracy']
)

model.summary()


# 4. Train the Model 
print("\n4. Training the CNN Model...")
history = model.fit(
    x_train, y_train_encoded,
    batch_size=128,
    epochs=10, # 10 epochs is sufficient for high accuracy 
    verbose=1,
    validation_data=(x_test, y_test_encoded)
)
print("   Training finished.")


# 5. Evaluate Performance 
print("\n5. Evaluating Model Performance on Test Set...")
score = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

# Check if the target accuracy is met 
if score[1] > 0.95:
    print("   Success! Test Accuracy target (>95%) achieved.")
else:
    print("   Note: Test Accuracy target not reached. Review model parameters.")


# 6. Visualize Predictions 
test_samples = x_test[:5]
test_labels = y_test[:5]
predictions = model.predict(test_samples)
predicted_classes = np.argmax(predictions, axis=1)

print("\n6. Visualizing 5 Sample Predictions (will require manual display of the plot).")

# Plotting the results 
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_samples[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {test_labels[i]}\nPred: {predicted_classes[i]}", fontsize=10)
    plt.axis('off')
plt.suptitle("CNN Predictions on Sample MNIST Images")
# In a real environment, you'd save this image for the report 
# plt.savefig('assets/cnn_predictions_sample.png')
# plt.show() # Uncomment to display plot

print("\nTask 2 completed. CNN trained and evaluated.")