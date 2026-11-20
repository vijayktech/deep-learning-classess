import zipfile
from PIL import Image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

with zipfile.ZipFile("Family_Face_Recognition.zip", "r") as zip_ref:
    zip_ref.extractall("./")

print("ZIP extracted successfully!")

image = Image.open("./family/Father/father1.jpg")

print("Original Image Size:", image.size)

# Load family photos from the folder
data = tf.keras.utils.image_dataset_from_directory(
    "./family",
    image_size=(64, 64),
    color_mode="rgb"
)

# Get class names from folder names
class_names = data.class_names

print("Classes:", class_names)

images = []
labels = []

# Get images and labels from dataset
for x, y in data:

    images.extend(x.numpy())
    labels.extend(y.numpy())

# Convert into NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize all image pixel values
images = images / 255.0

import numpy as np

images = []
labels = []

# Get images and labels from dataset
for x, y in data:

    images.extend(x.numpy())
    labels.extend(y.numpy())

# Convert into NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize all image pixel values
images = images / 255.0

# Create Sequential model
model = Sequential()

# Input: 64 rows, 64 columns, 3 RGB channels
model.add(Input(shape=(64, 64, 3)))

# Convolution: 4 filters of size 3x3
model.add(Conv2D(4, (3, 3), activation="relu"))

# Max Pooling: Reduce feature map size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten: Convert feature maps into 1D
model.add(Flatten())

# Hidden layer: 32 neurons
model.add(Dense(32, activation="relu"))

# Output layer: 3 family members
model.add(Dense(3, activation="softmax"))

# Display model structure
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(images, labels, epochs=20)

# Predict for uploading a image
image = tf.keras.utils.load_img(
    './family/Brother/brother5.jpg',
    target_size=(64, 64)
)

image = tf.keras.utils.img_to_array(image)

image = image / 255.0

image = np.array([image])

prediction = model.predict(image)


probabilities = list(prediction[0])

predicted_label = probabilities.index(max(probabilities))

print("Predicted Person:", class_names[predicted_label])