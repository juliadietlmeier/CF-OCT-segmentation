# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:47:34 2024

@author: dietlmj
"""
import keras
from keras import layers
from keras import ops
import tensorflow as tf
import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

# For data preprocessing
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return ops.nn.relu(x)




def squeeze_excite_block(self, x):
   		# store the input
    shortcut = x
   		# calculate the number of filters the input has
    filters = x.shape[-1]
   		# the squeeze operation reduces the input dimensionality
   		# here we do a global average pooling across the filters, which
   		# reduces the input to a 1D vector
    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
   		# reduce the number of filters (1 x 1 x C/r)
    x = layers.Dense(filters // self.ratio, activation="relu",
   			kernel_initializer="he_normal", use_bias=False)(x)
   		
   		# the excitation operation restores the input dimensionality
    x = layers.Dense(filters, activation="sigmoid",
    kernel_initializer="he_normal", use_bias=False)(x)
   		
   		# multiply the attention weights with the original input
    x = layers.Multiply()([shortcut, x])
   		# return the output of the SE block
    return x

def squeeze_excitation_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]
    
    # Squeeze phase
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation phase
    excitation = layers.Dense(channels // reduction_ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    
    # Reshape and scale
    #excitation = tf.reshape(excitation, (-1, 1, 1, channels))
    scaled_output = input_tensor * excitation
    
    return scaled_output

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    #x=dspp_input
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    out_1 = squeeze_excitation_block(dspp_input)
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    
    model_input = keras.Input(shape=(image_size, image_size, 3))#(image_size, image_size, 3)
    
    #preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    
    #resnet50 = keras.applications.ResNet50(
    #    weights="imagenet", include_top=False, input_tensor=preprocessed
    #)
    
    mobilenetv2 = keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_tensor=model_input, alpha=1.)
    
    #x = resnet50.get_layer("conv4_block6_2_relu").output
    x = mobilenetv2.get_layer("block_12_depthwise_relu").output
    
    
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    mobileLayers = {
            "shallowLayer": "block_2_project_BN", "deepLayer": "block_12_project_BN"}
    
    #input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = mobilenetv2.get_layer(mobileLayers["shallowLayer"]).output
    
    
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=512, num_classes=3)
model.summary()