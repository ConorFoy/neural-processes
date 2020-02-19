from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate, LSTM

from keras import backend as K

import tensorflow as tf
#import tensorflow_probability as tfp # for tf version 2.0.0, tfp version 0.8 is needed 


def simple_model(training_batch, window_size,
	lstm_units = 100,
	encoder_dropout = 0.1,
	decoder_dropout = 0.1):

	training_batch.contextify(window_size)

	context_shape = training_batch.context.shape # [batch_size,num_windows,timesteps_per_window,note_size]


	# ----------------- here define model

	input_context = Input(batch_shape = 
						  (context_shape[0],  # batch_size
						   context_shape[1],  # num_windows
						   context_shape[2],  # timesteps_per_window
						   context_shape[3]), # note_size
						   name="Input_layer_context") # as above
	
	input_target  = Input((None, 1), name="Input_layer_target")  # complete


	# Encoder

	encoder = input_context
	
	reshape_input_to_windows = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
									  name="Reshape_layer_1")(encoder)
	
	encoder = LSTM(units = lstm_units, dropout = encoder_dropout, name = 'Encoder_lstm')(reshape_input_to_windows)

	mean_representation = Lambda(lambda x: K.mean(tf.reshape(x, [context_shape[0], 
																 context_shape[1], 
																 lstm_units]), axis = -2),
								 name="Mean_representation_layer")(encoder)

	# Decoder

	propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1,150,1]))(mean_representation)

	decoder_input = propagate_in_time

	decoder, _, _ = LSTM(units = context_shape[3], 
				      dropout = encoder_dropout,
				      return_sequences = True,
				      return_state=True,
				      activation = 'sigmoid',
				      name = 'Decoder_lstm')(decoder_input)


	model = Model(input_context, decoder)

	#model = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
									    #name="Reshape_layer_1"),
								 #tf.keras.layers.LSTM(units = lstm_units, dropout = encoder_dropout, name = 'Encoder_lstm'),
								 #tf.keras.layers.Lambda(lambda x: K.mean(tf.reshape(x, [context_shape[0], 
																 #context_shape[1], 
																 #lstm_units]), axis = -2),
								 #name="Mean_representation_layer"),
								 #tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1,150,1])),
								 #tf.keras.layers.LSTM(units = context_shape[3], 
								      #dropout = encoder_dropout,
								      #return_sequences = True,
								      #return_state=True,
								      #activation = 'sigmoid',
								      #name = 'Decoder_lstm')
								 #])

	return model














