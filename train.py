import cv2
import numpy as np
from keras_squeezenet_tf2 import SqueezeNet
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "circleTurn": 0,
    "lim30": 1,
    "noP": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

'''
dataset = [
    [[...], 'rock'],
    [[...], 'paper'],
    ...
]
'''
data, labels = zip(*dataset)
labels = list(map(mapper, labels))


'''
labels: rock,paper,paper,scissors,rock...
one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
'''

# one hot encode the labels
labels = tf.keras.utils.to_categorical(labels,num_classes=4)
print(labels)
# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=10)

# save the model for later use
model.save("detect-sign-model.h5")
