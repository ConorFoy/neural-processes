from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate, LSTM

from keras import backend as K

import tensorflow as tf
#import tensorflow_probability as tfp # for tf version 2.0.0, tfp version 0.8 is needed 


def simple_model(training_batch,
	lstm_units = 512,
	encoder_dropout = 0.1,
	decoder_dropout = 0.1):

	context_shape = training_batch.context.shape # [batch_size,num_windows,timesteps_per_window,note_size]
	target_shape  = training_batch.target.shape  # [batch_size, timesteps, note_size]

	# ----------------- here define model

	input_context = Input(batch_shape = 
						  (context_shape[0],  # batch_size
						   context_shape[1],  # num_windows
						   context_shape[2],  # timesteps_per_window
						   context_shape[3]), # note_size
						  name="Input_layer_context") # as above
	
	input_target  = Input(batch_shape = 
						  (target_shape[0],
						   target_shape[1],
						   target_shape[2]), 
						  name="Input_layer_target")  


	# Encoder

	encoder = input_context
	
	reshape_input_to_windows = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
									  name="Reshape_layer_1")(encoder)
	
	encoder = LSTM(units = lstm_units, 
			       dropout = encoder_dropout, 
			       name = 'Encoder_lstm_1', 
			       return_sequences = True)(reshape_input_to_windows)
	
	encoder = LSTM(units = lstm_units, 
				   dropout = encoder_dropout, 
				   name = 'Encoder_lstm_2')(encoder)

	encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
	encoder = Dense(256, activation = 'relu', name = 'Encoder_dense_2')(encoder)
	encoder = Dense(10, activation = 'tanh', name = 'Encoder_dense_3')(encoder)

	mean_representation = Lambda(lambda x: K.mean(tf.reshape(x, [context_shape[0], 
																 context_shape[1], 
																 10]), axis = -2),
								 name="Mean_representation_layer")(encoder)

	# Decoder

	propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1,target_shape[1],1]))(mean_representation)

	decoder_input = Lambda(lambda x: tf.concat([input_target, propagate_in_time], axis = 2))

	decoder, _, _ = LSTM(units = lstm_units, 
				      dropout = decoder_dropout,
				      return_sequences = True,
				      return_state = True,
				      activation = 'tanh',
				      name = 'Decoder_lstm_1')(decoder_input)

	decoder, _, _ = LSTM(units = context_shape[3], 
				      dropout = decoder_dropout,
				      return_sequences = True,
				      return_state = True,
				      activation = 'sigmoid',
				      name = 'Decoder_lstm_2')(decoder)


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














