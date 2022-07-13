import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle

seed = "root folder which contains your images"

categories = [ x for x in os.listdir(f"./{seed}")]


def create_dataset(categories):
	label = 0
	images = []
	for cat in categories:
		if(cat == "no"):
			label = 0
		else:
			label = 1

		for image in os.listdir(f"./{seed}/{cat}"):
			img = cv2.resize(cv2.imread(f"./{seed}/{cat}/{image}", 0), (50, 50))
			img = img / 255.0

			images.append([img, label])

	shuffle(images)
	shuffle(images)

	return images

ds = create_dataset(categories)
X = []
y = []
for x,ys in ds:
	X.append(x), y.append(ys)

X = np.array(X)

np.save("save name", X)
np.save("save name", y)