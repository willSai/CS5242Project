import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Activation, MaxPool3D, Dropout, Flatten
from keras.layers import BatchNormalization
from keras import initializers
import numpy as np

def example_network(input_shape):

    im_input = Input(shape=input_shape)

    t = Conv3D(64, (11, 11, 11), padding='valid', kernel_initializer=initializers.truncated_normal(mean=0, stddev=0.001), bias_initializer=initializers.constant(0.1))(im_input)
    t = Activation('relu')(t)
    t = MaxPool3D(pool_size=(2, 2, 2), padding='valid')(t)

    t = Conv3D(128, (6, 6, 6), padding='valid', kernel_initializer=initializers.truncated_normal(mean=0, stddev=0.001), bias_initializer=initializers.constant(0.1))(t)
    t = Activation('relu')(t)
    t = MaxPool3D(pool_size=(2, 2, 2), padding='valid')(t)

    t = Conv3D(256, (3, 3, 3), padding="valid", kernel_initializer=initializers.truncated_normal(mean=0, stddev=0.001), bias_initializer=initializers.constant(0.1))(t)
    t = Activation('relu')(t)

    t = Flatten()(t)

    t = Dense(1000, kernel_initializer=initializers.truncated_normal(mean=0, stddev=1/np.sqrt(1000)), bias_initializer=initializers.constant(1.0))(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(500, kernel_initializer=initializers.truncated_normal(mean=0, stddev=1/np.sqrt(500)), bias_initializer=initializers.constant(1.0))(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(200, kernel_initializer=initializers.truncated_normal(mean=0, stddev=1/np.sqrt(200)), bias_initializer=initializers.constant(1.0))(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(1)(t)
    output = Activation('sigmoid')(t)

    model = Model(input=im_input, output=output)

    return model
