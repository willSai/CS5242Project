import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten

def example_network(input_shape=(28,28,1), class_num=10):
    """Example CNN
    
    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})
    
    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)
    
    t = Conv2D(16, (3,3))(im_input)
    t = Activation('relu')(t)
    t = MaxPool2D(pool_size=(2,2))(t)
    t = Dropout(0.9)(t)

    t = Flatten()(t)
    
    t = Dense(256)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)

    output = Activation('softmax')(t)

    model = Model(input=im_input, output=output)
    
    return model