import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization, Add
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.optimizers import RMSprop,adam
from tensorflow.python.keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.IntersectionOverUnion import bb_intersection_over_union
import numpy as np
import random
import os
import cv2

def load_model():
    input = Input(shape=(250,250,3))
    x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64,(3,3),use_bias=False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y = x
    x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64,(3,3),use_bias=False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y = x
    x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D((2,2))(x)
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    y = x
    x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y = x
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    y = x
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    y = x
    class_features = Flatten()(x)
    z = Dense(128,use_bias=False)(class_features)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    z = Dense(64,use_bias=False)(z)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    z = Dense(32,use_bias=False)(z)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    output_class = Dense(3,activation = 'softmax')(z)
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(512,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y = x
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    y = x
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    y = x
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    # x = Conv2D(255,(3,3),use_bias= False, padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    detect_features = Flatten()(x)
    # x = Dense(512,use_bias=False)(features)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    x = Dense(128,use_bias=False)(detect_features)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(64,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(32,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(32,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
#     output_class = Dense(3,activation = 'softmax')(x)
    output_regress = Dense(4)(x)
    network = Model(input,[output_class,output_regress])
#     network.summary()
    return network