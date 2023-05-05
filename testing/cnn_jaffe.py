import json
import os

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from testing.f1_score import F1

with open('results.json', 'r') as f:
    data = json.load(f)

train_data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
)

test_data = ImageDataGenerator(
    rescale=1./255,
)

train = train_data.flow_from_directory(
    directory='../datasets/jaffe/train',  # Directory with the pictures
    target_size=(256, 256),  # Reshape to this form
    batch_size=4,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,  # Shuffle the data on each load
    subset='training'  # Flag for training subset
)

# Generator for validation data
valid = train_data.flow_from_directory(
    directory='../datasets/jaffe/train', # Directory with the pictures
    target_size=(256, 256), # Reshape to this form
    batch_size=4,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,  # Shuffle the data on each load
    subset='validation'  # Flag for validation subset
)

# Generator for test data
test = test_data.flow_from_directory(
    '../datasets/jaffe/test',  # Directory with the pictures
    target_size=(256, 256), # Reshape to this form
    batch_size=4,
    color_mode='grayscale',
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1()])
model.fit(train, epochs=40, validation_data=valid)

test_loss, test_f1 = model.evaluate(test)
print(f'Test f1: {test_f1:.4f}, Test loss: {test_loss:.4f}')

if 'jaffe' not in data:
    data['jaffe'] = {}
data['jaffe']['CNN_accuracy'] = test_f1
data['jaffe']['CNN_params'] = model.count_params()

with open('results.json', 'w') as f:
    json.dump(data, f)

