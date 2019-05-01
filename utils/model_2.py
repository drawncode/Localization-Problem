import keras
from keras.models import Model
from keras.layers import Input,Add, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import os
import cv2
import imageio
import imgaug as ia
from imgaug import augmenters as iaa 
from utils.IntersectionOverUnion import bb_intersection_over_union


# %matplotlib inline
ia.seed(1)

def load_model():
    input = Input(shape=(700,700,3))
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
#     x = MaxPooling2D((2,2))(x)
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
#     x = MaxPooling2D((2,2))(x)
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    x = MaxPooling2D((2,2))(x)
    y = x
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
#     x = MaxPooling2D((2,2))(x)
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
#     x = MaxPooling2D((2,2))(x)
    y = x
    x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
#     x = MaxPooling2D((2,2))(x)
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
    x = MaxPooling2D((2,2))(x)
    y = x
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
#     x = MaxPooling2D((2,2))(x)
    x = Conv2D(512,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    x = MaxPooling2D((2,2))(x)
    y = x
    x = Conv2D(256,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512,(3,3),use_bias= False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Add()([x,y])
    features = Flatten()(x)
    x = Dense(128,use_bias=False)(features)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(64,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(32,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y=x
    x = Dense(32,use_bias=False)(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    output_1 = Dense(4)(x)
    x = Dense(32,use_bias=False)(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    output_2 = Dense(4)(x)
    x = Dense(32,use_bias=False)(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    output_3 = Dense(4)(x)
    x = Dense(32,use_bias=False)(y)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    output_4 = Dense(4)(x)
    network = Model(input,[output_1,output_2,output_3,output_4])
    return network