"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import keras.models
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Lambda, Conv2D, Dropout, Conv2DTranspose
from keras.layers import Concatenate, concatenate, Average

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))

    h = Dense(1024)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(1024)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(64)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(512)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(1024)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(1024)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)

    return Model(inputs=inputLayer, outputs=h)

def get_unet_model(inputDim, connect='avg'):

    input_layer = Input(shape=(inputDim,))

    h1 = Dense(128)(input_layer)
    t1 = BatchNormalization()(h1)
    t1 = Activation('relu')(t1)

    h2 = Dense(128)(t1)
    t2 = BatchNormalization()(h2)
    t2 = Activation('relu')(t2)

    h3 = Dense(128)(t2)
    t3 = BatchNormalization()(h3)
    t3 = Activation('relu')(t3)

    h4 = Dense(128)(t3)
    t4 = BatchNormalization()(h4)
    t4 = Activation('relu')(t4)

    h = Dense(8)(t4)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    if connect == 'avg':
        h5 = Dense(128)(h)
        h5 = Average()([h5, h4])
        h5 = BatchNormalization()(h5)
        h5 = Activation('relu')(h5)

        h6 = Dense(128)(h5)
        h6 = Average()([h6, h3])
        h6 = BatchNormalization()(h6)
        h6 = Activation('relu')(h6)

        h7 = Dense(128)(h6)
        h7 = Average()([h7, h2])
        h7 = BatchNormalization()(h7)
        h7 = Activation('relu')(h7)

        h8 = Dense(128)(h7)
        h8 = Average()([h8, h1])
        h8 = BatchNormalization()(h8)
        h8 = Activation('relu')(h8)
    else:
        h5 = Dense(128)(h)
        h5 = concatenate([h5, h4])
        h5 = BatchNormalization()(h5)
        h5 = Activation('relu')(h5)

        h6 = Dense(128)(h5)
        h6 = concatenate([h6, h3])
        h6 = BatchNormalization()(h6)
        h6 = Activation('relu')(h6)

        h7 = Dense(128)(h6)
        h7 = concatenate([h7, h2])
        h7 = BatchNormalization()(h7)
        h7 = Activation('relu')(h7)

        h8 = Dense(128)(h7)
        h8 = concatenate([h8, h1])
        h8 = BatchNormalization()(h8)
        h8 = Activation('relu')(h8)

    out = Dense(inputDim)(h8)

    return Model(inputs=input_layer, outputs=out)

#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

    
