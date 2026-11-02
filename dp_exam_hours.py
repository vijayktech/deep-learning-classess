# STEP 1: Import required libraries,
import numpy as np
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Input, Dense 
from keras import layers, models

# STEP 2: Create sample student data,
# Each record contains [Study Hours, Attendance],
# Attendance is written as a fraction (80% = 0.80),
# This keeps the input numbers in similar ranges.,
X = np.array([
    [1, 0.40],
    [2, 0.50],
    [3, 0.55],
    [4, 0.60],
    [5, 0.65],
    [6, 0.70],
    [7, 0.75],
    [8, 0.80],
    [9, 0.90],
    [10, 0.95],
], dtype=np.float32)

# Actual answers,
# 0 = Fail, 1 = Pass,
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# STEP 3: Create the neural network,
model = Sequential([
    # Input layer + first hidden layer (change input_shape to match your X columns)
    layers.Dense(units=8, activation="relu", input_shape=(X.shape[1],)), 
    
    # Hidden layer
    layers.Dense(units=4, activation="relu"),
    
    # Output layer (1 unit with sigmoid is standard for binary_crossentropy)
    layers.Dense(units=1, activation="sigmoid") 
])

# STEP 5: Compile the model,
model.compile(
    optimizer="adam",     # Updates weights and biases,
    loss="binary_crossentropy",    # Calculates classification loss,
    metrics=["accuracy"]   # Measures prediction accuracy,
)

# STEP 6: Train the model
model.fit(
    X,     # Input data
    y,     # Actual answers
    epochs=300    # Complete dataset processed 300 times
)

# STEP 7: Make a prediction for a new student
# Study Hours = 8, Attendance = 85%
new_student = np.array([[8, 0.85]], dtype=np.float32)

# Get the predicted probability
prediction = model.predict(new_student, verbose=0)

# Extract probability from the output array
probability = prediction[0][0]

print("Probability of passing:", probability)

# STEP 8: Convert probability into Pass or Fail
if probability >= 0.5:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
