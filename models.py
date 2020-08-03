from keras.models import Model, Sequential, load_model
from keras.layers import Flatten, Dense, Input, Lambda, Conv2D, MaxPooling2D, RepeatVector, Reshape
from keras.layers.merge import Concatenate, Multiply, Dot, Subtract
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
# from resnet50 import ResNet50

from keras import backend as K
# K.image_dim_ordering('th')
# K.common.image_dim_ordering()
import keras

# keras.backend.set_image_data_format('channels_first')
keras.backend.set_image_data_format('channels_last')

print(keras.backend.image_data_format())

def ObjectNet(object_dim):


	object_input = Input(shape = (object_dim, object_dim, 3), name = 'object_input')

	x = Conv2D(64, 3, activation='relu', padding = 'same', data_format='channels_last')(object_input)
	x = Conv2D(64, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = MaxPooling2D(pool_size = 2, data_format='channels_last')(x)

	# Block 2
	x = Conv2D(128, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(128, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = MaxPooling2D(pool_size = 2, data_format='channels_last')(x)

	# Block 3
	x = Conv2D(256, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(256, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(256, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = MaxPooling2D(pool_size = 2, data_format='channels_last')(x)

	# Block 4
	x = Conv2D(512, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = MaxPooling2D(pool_size = 2, data_format='channels_last')(x)

	# Block 5
	x = Conv2D(512, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = Conv2D(512, 3, activation='relu', padding = 'same', data_format='channels_last')(x)
	x = MaxPooling2D(pool_size = 2, data_format='channels_last')(x)

	x = Flatten()(x)
	x = Dense(4096, activation = 'relu')(x)
	x = Dense(1024, activation = 'relu', name = 'object_vector')(x)
	output = Dense(80, activation = 'softmax', name = 'objectnet_output')(x)

	model = Model(inputs = object_input, outputs = output)

	return model

def AddGist(objectnet, context_dim):

	k = 5
	context_input = Input(shape = (context_dim, context_dim, 3), name = 'context_input')
	x = Conv2D(64, k, activation='relu', strides = 2, padding='valid', data_format='channels_last')(context_input)
	x = Conv2D(64, k, activation='relu', strides = 2, padding='valid', data_format='channels_last')(x)
	x = Conv2D(128, k, activation='relu', strides = 2, padding='valid', data_format='channels_last')(x)
	x = Conv2D(128, k, activation='relu', strides = 2, padding='valid', data_format='channels_last')(x)
	x = Conv2D(256, k, activation='relu', strides = 2, padding='valid', data_format='channels_last')(x)
	x = Conv2D(256, 3, activation='relu', strides = 2, padding='valid', data_format='channels_last')(x)
	x = Conv2D(512, 3, activation='relu', padding='valid', data_format='channels_last')(x)
	x = Conv2D(512, 3, activation='relu', padding='valid', data_format='channels_last')(x)

	x = Flatten(name = 'gist_vector')(x)
	x = Concatenate()([x, objectnet.get_layer('object_vector').output])
	output = Dense(80, activation = 'softmax', name = 'gistnet_output')(x)

	model = Model(inputs = [objectnet.input, context_input], outputs = output)
	return model

def build_model(add, object_dim, **kwargs):

	if add == 'gist': return AddGist(ObjectNet(object_dim), kwargs['context_dim'])
	else: raise ValueError('Please specify a valid model.')
