import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,LeakyReLU,Dropout


##MODEL USAGE


def get_value(prediction):
	x = list(prediction[0])
	value = x.index(max(x))

	if value == 1:
		print("Tumor present:")
	else:
		print("no tumor present")


#x = np.load("datasets/features.npy")
#y = np.load("datasets/labels.npy")


image = np.array(cv2.resize(cv2.imread("image path", 0), (50, 50)))
image = image.reshape(-1,50,50,1)

model = keras.models.load_model("saved_model.h5")
prediction = model.predict(image)



get_value(prediction)