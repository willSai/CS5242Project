import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten

def example_network(input_shape=(21,21,21,19), class_num=10):
    """Example CNN
    Mind the 19 in input_shape, it refers to the number of our own features.
    """

    im_input = Input(shape=input_shape)
    
    t = Conv3D(64, (11,11,11))(im_input)
    t = Activation('relu')(t)
    t = MaxPool3D(pool_size=(3,3,3))(t)

    t = Conv3D(128, (6,6,6))(im_input)
    t = Activation('relu')(t)
    t = MaxPool3D(pool_size=(4,4,4))(t)

    t = Conv3D(256, (3,3,3))(im_input)
    t = Activation('relu')(t)
    t = MaxPool3D(pool_size=(5,5,5))(t)

    t = Flatten()(t)
    
    t = Dense(1000)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(500)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(200)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    # output = Activation('softmax')(t)

    model = Model(input=im_input, output=output)
    
    return model