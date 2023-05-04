import json
import os

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

# paths = ['C:\\Users\\user\\PycharmProjects\\GPImageClassification\\jaffe\\test\\happy',
#          'C:\\Users\\user\\PycharmProjects\\GPImageClassification\\jaffe\\test\\sad',
#          'C:\\Users\\user\\PycharmProjects\\GPImageClassification\\jaffe\\train\\happy',
#          'C:\\Users\\user\\PycharmProjects\\GPImageClassification\\jaffe\\train\\sad']
# for path in paths:
#     for filename in tqdm(os.listdir(path)):
#         path1 = os.path.join(path, filename)
#         plt.imsave(path1.replace('.tiff', '.png'), plt.imread(path1))
#         os.remove(path1)

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
    directory='../jaffe/train',  # Directory with the pictures
    target_size=(256, 256),  # Reshape to this form
    batch_size=4,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,  # Shuffle the data on each load
    subset='training'  # Flag for training subset
)

# Generator for validation data
valid = train_data.flow_from_directory(
    directory='../jaffe/train', # Directory with the pictures
    target_size=(256, 256), # Reshape to this form
    batch_size=4,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,  # Shuffle the data on each load
    subset='validation'  # Flag for validation subset
)

# Generator for test data
test = test_data.flow_from_directory(
    '../jaffe/test',  # Directory with the pictures
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=50, validation_data=valid)

test_loss, test_acc = model.evaluate(test)
print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')

if 'jaffe' not in data:
    data['jaffe'] = {}
data['jaffe']['CNN_accuracy'] = test_acc
data['jaffe']['CNN_params'] = model.count_params()

with open('results.json', 'w') as f:
    json.dump(data, f)

