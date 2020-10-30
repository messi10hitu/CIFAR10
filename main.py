import tensorflow as tf
from keras.constraints import maxnorm
from keras.utils import np_utils
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import SGD

data = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = data.load_data()
# print(train_images)
# print(train_labels)

# PREPROCESS THE DATA
# We do this to get the values between 0 and 1 to minimize the calculations
train_images = train_images.reshape(50000, 32, 32, 3)
train_images = train_images / 255.0
test_images = test_images.reshape(10000, 32, 32, 3)
test_images = test_images / 255.0

# train_labels = np_utils.to_categorical(train_labels)
# test_labels = np_utils.to_categorical(test_labels)
#
# num_classes = test_labels.shape[1]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# for i in range(9):
#     print(class_names[train_labels[i][0]])
#

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# BUILD THE MODEL
# Setup the layers:
# 1.input
# 2.hidden
# 3.output
"""
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),  # transforms the format of the images from a two-dimensional array
    # (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)

    keras.layers.Dense(128, activation="relu"),  # These are densely connected, or fully connected, neural layers.
    # The first Dense layer has 128 nodes (or neurons). This is the Hidden layer "relu" = rectified linear unit.
    # This hidden layer is used to manage our bias and the weights and figure out the patter for accurate results.

    keras.layers.Dense(10, activation="softmax")])  # The second (and last) layer returns a logits array with length
# of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
# This is the output layer which consist of 10 labels (0,9) and we wnt our output in the reangeB?w (0,1)

# SOFTMAX = The Softmax regression is a form of logistic regression that normalizes an input value
# into a vector of values that follows a probability distribution whose total sums up to 1.
# The output values are between the range [0,1].The function is usually used to compute losses
# that can be expected when training a data set


# compile the model:
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Optimizers ==> update the weight parameters to minimize the loss function
# This is how the model is updated based on the data it sees and its loss function.

# Loss function ==> measures how accurate the model is during training.
# You want to minimize this function to "steer" the model in the right direction.

# Metrics ==> Used to monitor the training and testing steps.

# TRAIN THE MODEL:

# feed the model:
model.fit(train_images, train_labels, epochs=120, validation_data=(test_images, test_labels))
# epochs tells us how many times the model is going to see the (train_images, train_labels) to get the better accuracy

# evaluate the accuracy:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("loss: ", test_loss)
print("Acurracy: ", test_acc)

# save the model
model.save("CNN_IMAGE_model.h5")
"""

# Load the model
model = keras.models.load_model('CNN_IMAGE_model.h5')
model.summary()

# prediction = model.predict(test_images)
# print(prediction)
# print(len(prediction))
# print("-----------------------")
# print(prediction[0])
# print(np.argmax(prediction[0]))
# # print("----------------")
#
# for i in range(30):
#     print("actual: ", class_names[test_labels[i][0]])
#     print("prediction:", class_names[np.argmax(prediction[i])])
#
# # for i in range(5):
# #     plt.grid(False)
# #     plt.imshow((np.squeeze(test_images[i])), cmap=plt.cm.binary)
# #     plt.xlabel("Actual: " + class_names[test_labels[i][0]])
# #     plt.title("Prediction:" + class_names[np.argmax(prediction[i])])
# #     plt.show()
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow((np.squeeze(test_images[i])), cmap=plt.cm.binary)
#     plt.xlabel("Actual: " + class_names[test_labels[i][0]])
#     # plt.title("Prediction:" + class_names[np.argmax(prediction[i])])
# plt.show()
#
results = {
    0: 'aeroplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
from PIL import Image

im = Image.open("./images/animal-animal-photography-close-up-countryside-1996333.jpg")
# the input image is required to be in the shape of dataset, i.e (32,32,3)

im = im.resize((32, 32))
im = np.expand_dims(im, axis=0)
im = np.array(im)
im = im / 255
pred = model.predict_classes([im])[0]
print(pred, results[pred])
#
