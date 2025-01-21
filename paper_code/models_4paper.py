#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:19:45 2024

@author: daa
"""


#import skimage
import time
import os
#import scipy
#from scipy import ndimage as ndi
#from skimage.transform import resize
#from skimage.io import imsave
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as keras
#from keras.preprocessing.image import ImageDataGenerator
#import imageio
import keras.backend as K
from keras.applications.efficientnet import EfficientNetB4
#import efficientnet.keras as efn

from keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications import VGG16

class WeightedSCCE(tf.keras.losses.Loss):
        def __init__(self, class_weight, from_logits=False, name='weighted_scce'):
            if class_weight is None or all(v == 1. for v in class_weight):
                self.class_weight = None
            else:
                self.class_weight = tf.convert_to_tensor(class_weight,
                    dtype=tf.float32)
            self.name = name
            self.reduction = tf.keras.losses.Reduction.NONE
            self.unreduced_scce = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=from_logits, name=name,
                reduction=self.reduction)
    
        def __call__(self, y_true, y_pred, sample_weight=None):
            loss = self.unreduced_scce(y_true, y_pred, sample_weight)
            if self.class_weight is not None:
                self.class_weight=tf.cast(self.class_weight, tf.float32)
                y_true=tf.cast(y_true, tf.int32)
                weight_mask = tf.gather(self.class_weight, y_true)
                loss = tf.math.multiply(loss, weight_mask)
            
            return loss

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


ki=tf.keras.initializers.he_normal(seed=0)
ng=2
dropout=0.2

def unet1024_GN(pretrained_weights=None, input_shape=(1024, 1024, 3), num_classes=3):
    inputs = Input(shape=input_shape)
    
    down0b = Conv2D(8, (3,3),padding='same', activation='relu', kernel_initializer=ki)(inputs)
    #down0b = BatchNormalization()(down0b)
    down0b = GroupNormalization(groups=ng)(down0b)

    down0b = Conv2D(8, (3,3),padding='same', activation='relu', kernel_initializer=ki)(down0b)
    #down0b = BatchNormalization()(down0b)
    down0b = GroupNormalization(groups=ng)(down0b)
    #down0b = Dropout(0.5)(down0b)
    pool0b = MaxPooling2D((2,2), strides=(2,2))(down0b)
    
    down0a = Conv2D(16, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool0b)
    #down0a = BatchNormalization()(down0a)
    down0a = GroupNormalization(groups=ng)(down0a)
    down0a = Conv2D(16, (3,3),padding='same', activation='relu', kernel_initializer='he_normal')(down0a)
    #down0a = BatchNormalization()(down0a)
    down0a = GroupNormalization(groups=ng)(down0a)
    #down0a = Dropout(0.5)(down0a)
    pool0a = MaxPooling2D((2,2), strides=(2,2))(down0a)
    
    down0 = Conv2D(32, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool0a)
    #down0 = BatchNormalization()(down0)
    down0 = GroupNormalization(groups=ng)(down0)
    down0 = Conv2D(32, (3,3),padding='same', activation='relu', kernel_initializer=ki)(down0) 
    #down0 = BatchNormalization()(down0)
    down0 = GroupNormalization(groups=ng)(down0)
    #down0 = Dropout(0.5)(down0)
    pool0 = MaxPooling2D((2,2), strides=(2,2))(down0)
    
    down1 = Conv2D(64, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool0)
    #down1 = BatchNormalization()(down1)
    down1 = GroupNormalization(groups=ng)(down1)
    down1 = Conv2D(64, (3,3),padding='same', activation='relu', kernel_initializer=ki)(down1)
    #down1 = BatchNormalization()(down1)
    down1 = GroupNormalization(groups=ng)(down1)
    #down1 = Dropout(0.5)(down1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(down1)
    
    down2 = Conv2D(128, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool1)
    #down2 = BatchNormalization()(down2)
    down2 = GroupNormalization(groups=ng)(down2)
    down2 = Conv2D(128, (3,3),padding='same', activation='relu', kernel_initializer=ki)(down2)
    #down2 = BatchNormalization()(down2)
    down2 = GroupNormalization(groups=ng)(down2)
    #down2 = Dropout(0.5)(down2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(down2)
    	
    down3 = Conv2D(256, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool2)
    #down3 = BatchNormalization()(down3)
    down3 = GroupNormalization(groups=ng)(down3)
    down3 = Conv2D(256, (3,3),padding='same', activation='relu', kernel_initializer=ki)(down3)
    #down3 = BatchNormalization()(down3)
    down3 = GroupNormalization(groups=ng)(down3)
    #down3 = Dropout(0.5)(down3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(down3)
    
    down4 = Conv2D(512, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool3)
    #down4 = BatchNormalization()(down4)
    down4 = GroupNormalization(groups=ng)(down4)
    down4 = Conv2D(512, (3,3),padding='same', activation='relu', kernel_initializer=ki)(down4)
    #down4 = BatchNormalization()(down4)
    down4 = GroupNormalization(groups=ng)(down4)
    #down4 = Dropout(0.5)(down4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(down4)
#---------------------------------------------------------------------------------------------------------------------------    
    center = Conv2D(1024, (3,3),padding='same', activation='relu', kernel_initializer=ki)(pool4)
    #center = BatchNormalization()(center)
    center = GroupNormalization(groups=ng)(center)
    center = Conv2D(1024, (3,3),padding='same', activation='relu', kernel_initializer=ki)(center)
    #center = BatchNormalization()(center)
    center = GroupNormalization(groups=ng)(center)
#---------------------------------------------------------------------------------------------------------------------------
    up4 = Conv2D(512, (3,3),padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(center))
    #up4 = BatchNormalization()(up4)
    up4 = GroupNormalization(groups=ng)(up4)
    merge4 = concatenate([down4, up4], axis = 3)
    up4 = Conv2D (512, (3,3),padding='same', activation='relu', kernel_initializer=ki)(merge4)
    #up4 = BatchNormalization()(up4)
    up4 = GroupNormalization(groups=ng)(up4)
    up4 = Conv2D (512, (3,3),padding='same', activation='relu', kernel_initializer=ki)(up4)
    #up4 = BatchNormalization()(up4)
    up4 = GroupNormalization(groups=ng)(up4)
    up4 = Dropout(dropout)(up4)
#------------------------------------------------------------------------------    
    up3 = Conv2D(256, (3,3),padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(up4))
    #up3 = BatchNormalization()(up3)
    up3 = GroupNormalization(groups=ng)(up3)
    merge3 = concatenate([down3,up3], axis=3)
    up3 = Conv2D(256, (3,3),padding='same', activation='relu', kernel_initializer=ki)(merge3)
    #up3 = BatchNormalization()(up3)
    up3 = GroupNormalization(groups=ng)(up3)
    up3 = Conv2D(256, (3,3),padding='same', activation='relu', kernel_initializer=ki)(up3)
    #up3 = BatchNormalization()(up3)
    up3 = GroupNormalization(groups=ng)(up3)
    up3 = Dropout(dropout)(up3)
#------------------------------------------------------------------------------    
    up2 = Conv2D(128, (3,3),padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(up3))
    #up2 = BatchNormalization()(up2)
    up2 = GroupNormalization(groups=ng)(up2)
    merge2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3,3),padding='same', activation='relu', kernel_initializer=ki)(merge2)
    #up2 = BatchNormalization()(up2)
    up2 = GroupNormalization(groups=ng)(up2)
    up2 = Conv2D(128, (3,3),padding='same', activation='relu', kernel_initializer=ki)(up2)
    #up2 = BatchNormalization()(up2)
    up2 = GroupNormalization(groups=ng)(up2)
    up2 = Dropout(dropout)(up2)
#------------------------------------------------------------------------------    
    up1 = Conv2D(64, (3,3),padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(up2))
    #up1 = BatchNormalization()(up1)
    up1 = GroupNormalization(groups=ng)(up1)
    merge1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3,3),padding='same', activation='relu', kernel_initializer=ki)(merge1)
    #up1 = BatchNormalization()(up1)
    up1 = GroupNormalization(groups=ng)(up1)
    up1 = Conv2D(64, (3,3),padding='same', activation='relu', kernel_initializer=ki)(up1)
    #up1 = BatchNormalization()(up1)
    up1 = GroupNormalization(groups=ng)(up1)
    up1 = Dropout(dropout)(up1)
#------------------------------------------------------------------------------    
    up0 = Conv2D(32, (3,3),padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(up1))
    
    #up0 = BatchNormalization()(up0)
    up0 = GroupNormalization(groups=ng)(up0)
    merge0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3,3),padding='same', activation='relu', kernel_initializer=ki)(merge0)
    #up0 = BatchNormalization()(up0)
    up0 = GroupNormalization(groups=ng)(up0)
    up0 = Conv2D(32, (3,3),padding='same', activation='relu', kernel_initializer=ki)(up0)
    #up0 = BatchNormalization()(up0)
    up0 = GroupNormalization(groups=ng)(up0)
    up0 = Dropout(dropout)(up0)
#------------------------------------------------------------------------------
    up0a = Conv2D(16, (3,3),padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(up0))
    #up0a = BatchNormalization()(up0a)
    up0a = GroupNormalization(groups=ng)(up0a)
    merge0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3,3),padding='same', activation='relu', kernel_initializer=ki)(merge0a)
    #up0a = BatchNormalization()(up0a)
    up0a = GroupNormalization(groups=ng)(up0a)
    up0a = Conv2D(16, (3,3),padding='same', activation='relu', kernel_initializer=ki)(up0a)
    #up0a = BatchNormalization()(up0a)
    up0a = GroupNormalization(groups=ng)(up0a)
    up0a = Dropout(dropout)(up0a)
#------------------------------------------------------------------------------    
    up0b = Conv2D(8, (3,3), padding='same', activation='relu', kernel_initializer=ki)(UpSampling2D(size=(2,2))(up0a))
    #up0b = BatchNormalization()(up0b)
    up0b = GroupNormalization(groups=ng)(up0b)
    merge0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3,3), padding='same', activation='relu', kernel_initializer=ki)(merge0b)
    #up0b = BatchNormalization()(up0b)
    up0b = GroupNormalization(groups=ng)(up0b)
    up0b = Conv2D(8, (3,3) ,padding='same', activation='relu', kernel_initializer=ki)(up0b)
    #up0b = BatchNormalization()(up0b)
    up0b = GroupNormalization(groups=ng)(up0b)
    up0b = Dropout(dropout)(up0b)
#------------------------------------------------------------------------------    
    classify = Conv2D(num_classes, (1,1), activation='softmax')(up0b)
    
    model = Model(inputs=inputs, outputs=classify)
    
#------------------------------------------------------------------------------    
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def dice_coef2(y_true, y_pred, smooth=1e-10):#1e-7
        #y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,0:])#was 1 [Ellipsis,1:])
        y_true_f = K.flatten(y_true[...,0:])
        y_pred_f = K.flatten(y_pred[...,0:])#was 1 maybe because background was not included?
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))
    
    def wscce_loss(y_true,y_pred):
        return WeightedSCCE(y_true,y_pred)
    
    def dice_loss(y_true, y_pred):
        return 1 - dice_coef2(y_true, y_pred)

#------------------------------------------------------------------------------
    
    #opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    #metrics = [dice_coef2, 'accuracy']
    #model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)#dice_loss#WeightedSCCE([1,1,1,1]) 'sparse_categorical_crossentropy'
    # labels can be integer if using sparse_categorical_crossentropy  
    #if(pretrained_weights):
    #    model.load_weights(pretrained_weights)
        
    return model



def UEfficientNet(input_shape=(None, None, 3), dropout_rate=0.2):

    backbone = EfficientNetB4(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.get_layer('block7b_dwconv').output#backbone.layers[342].output
    #conv4 = backbone.layers[342].output
    
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons * 64, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 64)
    convm = residual_block(convm,start_neurons * 64)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 32, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 32)
    uconv4 = residual_block(uconv4,start_neurons * 32)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.get_layer('block5a_dwconv').output#backbone.layers[154].output
    #conv3 = backbone.layers[154].output
    
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(dropout_rate)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 16)
    uconv3 = residual_block(uconv3,start_neurons * 16)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv3)
    #conv2 = backbone.get_layer('block4a_expand_conv').output#backbone.layers[92].output
    conv2 = backbone.get_layer('block3a_dwconv').output#backbone.layers[92].output
    #conv2 = backbone.layers[92].output
    
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 8)
    uconv2 = residual_block(uconv2,start_neurons * 8)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.get_layer('block2a_dwconv').output#backbone.layers[30].output
    #conv1 = backbone.layers[30].output
    
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 4)
    uconv1 = residual_block(uconv1,start_neurons * 4)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    
    deconv0 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv1)
    conv0 = backbone.get_layer('block1a_dwconv').output#backbone.layers[30].output
    #conv0 = backbone.layers[30].output
    
    uconv0 = concatenate([deconv0, conv0])
    
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 2)
    uconv0 = residual_block(uconv0, start_neurons * 2)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    #uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    #uconv0 = Dropout(0.1)(uconv0)
    #uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    #uconv0 = residual_block(uconv0,start_neurons * 1)
    #uconv0 = residual_block(uconv0,start_neurons * 1)
    #uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    uconv0=UpSampling2D(size=(2,2))(uconv0)
    output_layer = Conv2D(4, (1,1), padding="same", activation="softmax")(uconv0)    
    
    model = Model(input, output_layer)
    #model.name = 'u-xception'

    return model





def conv_block_vgg16(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block_vgg16(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block_vgg16(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block_vgg16(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block_vgg16(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block_vgg16(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block_vgg16(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(4, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model