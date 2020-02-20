#!../testenv/bin/python3

import pretty_midi
import numpy as np
#import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate, LSTM

from keras import backend as K

import tensorflow as tf
#import tensorflow_probability as tfp # for tf version 2.0.0, tfp version 0.8 is needed 
import numpy as np

import csv
from sys import stdout
import random
from datetime import date
# My code
from loading import *
from models import *


def generate(train_batch):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        new_batch = next(train_batch)
        yield (new_batch.context, new_batch.target)

if __name__ == '__main__':

	print("TensorFlow version: {}".format(tf.__version__))
	print("GPU is available: {}".format(tf.test.is_gpu_available()))

	file = 'maestro-v2.0.0/maestro-v2.0.0.csv'

	# Call data class
	data = DataObject(file, what_type = 'train', train_sec = 15, test_sec = 5, fs = 50, window_size = 15)

	# Create a batch class which we will iterate over
	train_batch = Batch(data, batch_size = 64, songs_per_batch = 4)

	curr_batch = train_batch.data
	model = simple_model(curr_batch)
	model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = 'adam')

	model.summary()

	history = model.fit_generator(
                    generate(train_batch),
                    steps_per_epoch=1024,
                    epochs=10)


	filename = date.today()

	# dd/mm/YY
	filename = 'training_'+filename.strftime("%d/%m/%Y")+'.txt'

	model.save_weights(filename+'.h5')

	with open(filename, 'w+') as f:
		for element in history.history['loss']:
			f.write('\n'+element)


