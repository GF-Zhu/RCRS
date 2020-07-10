# -*- encoding: utf-8 -*-

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras import backend as keras
from keras.utils import plot_model
from keras.layers import concatenate
import cmath
import keras.backend as K


def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def r_square(y_true, y_pred):
    r2 = 1 - K.sum(K.square(y_true - y_pred),axis=-1) / K.sum(K.square(y_true-K.mean(y_true)),axis=-1)
    return r2



def solar_center_radius_cnn(pretrained_weights = None,input_size = (512,512,1), is_binary_label=False):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  #64 * 256 *256

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)  
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  #128 * 128 * 128

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  #256 * 64 * 64

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    conv4 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    if is_binary_label:
        flatten1 = Flatten()(conv4)
        dense1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flatten1)
        dense1 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense1)
        dense1 = Dense(1536)(dense1)
    else:
        flatten1 = Flatten()(conv4)
        dense1 = Dense(256 , activation= 'relu',  kernel_regularizer = regularizers.l2(0.01))(flatten1)
        dense1 = Dense(16, activation='relu',  kernel_regularizer = regularizers.l2(0.01))(dense1)
        dense1 = Dense(3)(dense1)

    model = Model(input=inputs, output=dense1)

    optimizer = Adam(lr=1e-4)  
    lr_metric = get_lr_metric(optimizer)

    if is_binary_label:
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc', lr_metric])  #
    else:
        model.compile(optimizer = optimizer, loss = 'MSE',  metrics = [r_square, lr_metric])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights, by_name=True)
    # for layer in model.layers[:14]:
    #     layer.trainable = False

    return model


