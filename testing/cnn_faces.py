import json
from keras import backend as K
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# paths = ['C:\\Users\\user\\PycharmProjects\\GPImageClassification\\faces1\\test\\0',
#          'C:\\Users\\user\\PycharmProjects\\GPImageClassification\\faces1\\test\\non-0',
#          'C:\\Users\\user\\PycharmProjects\\GPImageClassification\\faces1\\train\\0',
#          'C:\\Users\\user\\PycharmProjects\\GPImageClassification\\faces1\\train\\non-0']
# for path in paths:
#     for filename in tqdm(os.listdir(path)):
#         path1 = os.path.join(path, filename)
#         plt.imsave(path1.replace('.pgm', '.png'), plt.imread(path1))
#         os.remove(path1)

def f1_score(y_true, y_pred):
    """
    Computes the F1 score given the true and predicted labels.
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score



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
    directory='../faces1/train',  # Directory with the pictures
    target_size=(19, 19),  # Reshape to this form
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,  # Shuffle the data on each load
    subset='training'  # Flag for training subset
)

# Generator for validation data
valid = train_data.flow_from_directory(
    directory='../faces1/train',  # Directory with the pictures
    target_size=(19, 19),  # Reshape to this form
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,  # Shuffle the data on each load
    subset='validation'  # Flag for validation subset
)

# Generator for test data
test = test_data.flow_from_directory(
    '../faces1/test',  # Directory with the pictures
    target_size=(19, 19),  # Reshape to this form
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score])
model.fit(train, epochs=15, validation_data=valid)

test_loss, test_acc = model.evaluate(test)
test_loss, test_acc = round(test_loss, 4), round(test_acc, 4)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

if 'faces' not in data:
    data['faces'] = {}
data['faces']['CNN_accuracy'] = test_acc
data['faces']['CNN_params'] = model.count_params()

with open('results.json', 'w') as f:
    json.dump(data, f)
