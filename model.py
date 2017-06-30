import numpy as np
np.random.seed(42)

from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

import json

class Model(object):

	def restore(self, path="model.json"):		
		self.model = Sequential()
		# self.model = model_from_json(json.load(open("model.json")))
		self.model = model_from_json(json.load(open(path)))
		
		# self.model.load_weights("model.h5")
		self.model.load_weights(path.replace('json', 'h5'))

		print("MODEL Restored")
		return self

	def predict(self, img, frame=[0]):
		frame[0]+=1
		print("current frame: " + str(frame[0]))
		steering_angle = self.model.predict(np.expand_dims(img, axis=0))[0][0]
		return steering_angle

	def model_setup(self):
		print("Setting up the model")
		# ================================================
		image_size = (160, 320, 3) # Same as source from emulator
		# ================================================
		# CommaAI Model:
		# https://github.com/commaai/research/blob/master/train_steering_model.py
		model = Sequential()
		model.add(Lambda(lambda x: x/127.5 - 1.,
						input_shape=image_size,
						output_shape=image_size))
		model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
		# -- Experimenting with MaxPooling:
		# model.add(Convolution2D(16, 8, 8, border_mode="same"))
		# model.add(MaxPooling2D(pool_size=(4, 4), border_mode='same'))
		# --
		model.add(ELU())
		model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
		model.add(ELU())
		model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
		model.add(Flatten())
		model.add(Dropout(.2))
		model.add(ELU())
		model.add(Dense(512))
		model.add(Dropout(.5))
		model.add(ELU())
		model.add(Dense(1))
		# ================================================	
		self.model = model
		
		# self.loss_fn = "mse"
		# optimizer = Adam(lr=0.001)
		self.model.compile(loss="mse", optimizer=Adam(lr=0.0001))
		# self.model.compile(loss="mse", optimizer=Adam(lr=0.001))
		# self.model.compile(loss="mse", optimizer=Adam(lr=0.01))
		# self.model.compile(loss="mse", optimizer=Adam(lr=0.03))

		return self.model
