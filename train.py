from model import Model

import os
import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageOps

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import json

# Load training data
data = []
# Each folder in "data/" dir contains "driving_log.csv" and subfolder "IMG/"
for f in os.scandir('data'):
    if f.is_dir():
        print("Processing data folder: " + str(f.path))
        # Append each pair of 'driving_log.csv' and 'IMG' to 'data' array
        data.append(( str(f.path) + "/driving_log.csv", str(f.path) + "/IMG" ))

# Processing driving_logs
driving_logs = []
for log_path, image_folder in data:
	log = pd.read_csv(log_path, header=None, names=["CenterImage", "LeftImage", "RightImage", "SteeringAngle", "Throttle", "Break", "Speed"])

	log["CenterImage"] = log["CenterImage"].str.rsplit("/", n=1).str[-1].apply(lambda p: os.path.join(image_folder, p))
	log["LeftImage"] = log["LeftImage"].str.rsplit("/", n=1).str[-1].apply(lambda p: os.path.join(image_folder, p))
	log["RightImage"] = log["RightImage"].str.rsplit("/", n=1).str[-1].apply(lambda p: os.path.join(image_folder, p))

	# print(log)
	driving_logs.append(log)

log = pd.concat(driving_logs, axis=0, ignore_index=True)


# Create new column of log["CenterImage"] size
log["CenterIMG"] = np.empty(log["CenterImage"].shape[0],dtype=object)

# ==================================================================
# ==================================================================
# Mirroring part of the images
# [DISABLED] – Too slow on big dataset..
# ==================================================================
# # Create mirror of log
# mirror = log.copy()

# # ----------------------------------------------------------------
# # Load images and process mirrors
# for i, row in log["CenterImage"].iteritems():	
# 	# Open image
# 	img = load_img(log["CenterImage"][i], target_size=(160, 320))
# 	# Convert to array
# 	img = np.asarray(img, dtype=np.uint8)
# 	# set value to 'image as array'
# 	log.set_value( i, 'CenterIMG', img.astype(np.float32) )
# 	log.set_value( i, 'SteeringAngle', log["SteeringAngle"][i].astype(np.float32) )
# 	# Reverse the steering
# 	mirror.set_value(i, 'SteeringAngle', - mirror["SteeringAngle"][i].astype(np.float32) )
# 	# Mirror the center image
# 	mirror.set_value(i, 'CenterIMG', img[:, ::-1, :].astype(np.float32))
# # ----------------------------------------------------------------

# # Combine origina 'log' array with 'mirror'
# log = pd.concat([log, mirror], axis=0, ignore_index=True)
# ==================================================================
# ==================================================================


# Shuffle dataframe ---------------------
# log = log.sample(frac=1)
# http://stackoverflow.com/questions/15772009/shuffling-permutating-a-dataframe-in-pandas
log = log.reindex(np.random.permutation(log.index))
print("log.shape", log.shape)
# ---------------------------------------


# Train | Valid --------------------------
# 15% of dataset wil be used for validation
validation_log = log[:int(log.shape[0]*.15)]
# the rest 85% is used for training
training_log = log[int(log.shape[0]*.15):]
print("training_log.shape", training_log.shape)
print("validation_log.shape", validation_log.shape)
# ---------------------------------------


# Callbacks -----------------------------
# Keras EarlyStopping callback (https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L319)
# Stop training when a 'val_loss' has stopped improving
stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

# Keras ModelCheckpoint callback (https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L220)
# Save the model after every epoch
filepath = "weights-improvement.{epoch:02d}-{val_loss:.2f}.hdf"
checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
# ---------------------------------------


# ===============
#batch_size: 32 -> 256
def createBatchGenerator(driving_log, batch_size=256): 
	batch_images = np.zeros((batch_size, 160, 320, 3))
	batch_steering = np.zeros(batch_size)
	while 1:
		for i in range(int(batch_size/2)):
			sample = log.sample(n=1)

			# img = load_img(log["CenterImage"][i], target_size=(160, 320))
			img = load_img(sample.iloc[0]["CenterImage"], target_size=(160, 320))
			img = np.asarray(img, dtype=np.uint8)
			batch_images[i] = img.astype(np.float32)
			batch_steering[i] = sample.iloc[0]['SteeringAngle']
			

			# === using different sample image for mirror ===|
			sample = log.sample(n=1)
			img = load_img(sample.iloc[0]["CenterImage"], target_size=(160, 320))
			img = np.asarray(img, dtype=np.uint8)
			# ===|

			batch_images[int(i+batch_size/2)] = img[:, ::-1, :].astype(np.float32)
			batch_steering[int(i+batch_size/2)] = - sample.iloc[0]['SteeringAngle']
		
		yield (batch_images, batch_steering)
# ===============


# Import the backend module
from keras import backend as K

# Sets the value of the image dimension ordering convention for tensorflow ('tf').
K.set_image_dim_ordering("tf")

# Training the model
with tf.Session() as sess:
	try: # Used to catch "KeyboardInterrupt" and "SystemExit" to save weights and model before exit. Useful for fast debugging. 

		nb_epoch = 10

		# create the model
		K.set_session(sess)

		model = Model()
		model = model.model_setup()

		# model.fit_generator(createBatchGenerator(log), samples_per_epoch = 102400, nb_epoch = nb_epoch, verbose=1)
		# fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=[], validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=False)

		# nb_val_samples – number of samples to use from validation generator at the end of every epoch. (50% from valid dataset).
		if validation_log.shape[0] < 1024:
			nb_val_samples = int(validation_log.shape[0]*0.5) # Use half of validation_log
		else:
			nb_val_samples = 1024 # Use 1024 datasamles from validation_log
		print("nb_val_samples: ", nb_val_samples)

		# model.fit_generator(createBatchGenerator(training_log), samples_per_epoch = 102400, validation_data=createBatchGenerator(validation_log), nb_val_samples=nb_val_samples,  nb_epoch = nb_epoch, verbose=1)

		model.fit_generator(createBatchGenerator(training_log), samples_per_epoch = 1024, validation_data=createBatchGenerator(validation_log), nb_val_samples=nb_val_samples,  nb_epoch = nb_epoch, verbose=1, callbacks=[stop_callback, checkpoint_callback])


		# model.save("model")
		json.dump(model.to_json(), open("model.json", "w"))
		model.save_weights("model.h5")


	except (KeyboardInterrupt, SystemExit): # Used to catch "KeyboardInterrupt" and "SystemExit" to save weights and model before exit.
		print("==============================================================================")
		print("Training Interrupted!")
		print("Saving model..")
		json.dump(model.to_json(), open("model.json", "w"))
		model.save_weights("model.h5")	    

