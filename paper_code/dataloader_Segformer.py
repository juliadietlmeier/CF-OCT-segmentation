#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:34:16 2024

@author: daa
"""

import cv2
import os 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow import keras
#import keras
from tensorflow.keras import backend
import pandas as pd







mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])

def normalize(input_image):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    #input_mask -= 1
    return input_image


import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    """Returns a TF Dataset."""

    def load_img_masks(input_img_path, target_img_path):
        
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        
        #input_img = tf.transpose(input_img, perm=[2,0,1])
        print('input_img shape = ', np.shape(input_img))
        
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")
        input_img = normalize(input_img/255)
        input_img = tf.transpose(input_img, perm=[2,0,1])
        
        #input_img = keras.layers.Permute(dims=(0,1,2))(input_img)
        #input_img=tf.stack(input_img,input_img,input_img)
        print(np.shape(input_img))
        
        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")
        
        target_img = tf.squeeze(target_img,2)
        #target_img = tf.transpose(target_img, perm=[2,0,1])
        
        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        #target_img -= 1
        return input_img, target_img

    # For faster debugging, limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
        
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    
    return dataset.batch(batch_size)




class My_Dataset:
    """OCT Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['fibrosis', 'ci_track', 'st_free_space', 'background']





    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [0,1,2,3] #[self.CLASSES.index(cls.lower()) for cls in classes]
        print(self.class_values)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image,(512,512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#--- rescale ------------------------------------------------------------------        
        #image = image/255# for UNET and VGG16
        #image = image # for Efficient Net no rescaling is needed
#------------------------------------------------------------------------------ 
        #print('image size = ', np.shape(image))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask,(512,512))
        
        #image,mask = normalize(image,mask)
        old_mask=mask
        #print('np.unique(mask)=',np.shape(mask))
        #mask = mask/255
        #print('mask shape = ', np.shape(mask))
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #print(self.class_values)
        #mask = np.stack(masks, axis=-1).astype('float')
        
        
        new_mask = np.zeros(mask.shape + (4,))
        #---- one-hot encoding ------------------------------------------------
        new_mask[mask == 0.,   0] = 1
        new_mask[mask == 1.,   1] = 1
        new_mask[mask == 2.,   2] = 1
        new_mask[mask == 3.,   3] = 1
        
        
        id2label = {0: "background", 1: "ST_free_space", 2: "CI_track", 3: "Fibrosis"}
        
        color2id = {
            0: 0,  # background pixel
            1: 1,  # Blue - Stomach
            2: 2,  # Green - Small bowel
            3: 3,  # Red - Large bowel
            }
 
# Reverse map from id to color
        id2color = {v: k for k, v in color2id.items()}
        
        #mask = new_mask#
        def rgb_to_onehot_to_gray(rgb_arr, color_map=id2color):
            num_classes = len(color_map)
            shape = rgb_arr.shape[:2] + (num_classes,)
            arr = np.zeros(shape, dtype=np.float32)
 
            for i, cls in enumerate(color_map):
                arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_map[i], axis=1).reshape(shape[:2])
 
            return arr.argmax(-1)
        
        mask = rgb_to_onehot_to_gray(mask, color_map=id2color)
        print('mask shape = ', np.shape(mask))
        #----------------------------------------------------------------------
        #for i in range(4):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            
        #    new_mask[mask == i,i] = 1
        #    print('new_mask shape = ', np.shape(new_mask))
        #new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3]))
        #mask = new_mask
        
        
        
        #print(np.unique(mask))
        # add background if mask is not binary
        #print(mask.shape[-1])
        #if mask.shape[-1] != 1:
        #    background = 1 - mask.sum(axis=-1, keepdims=True)
        #    mask = np.concatenate((mask, background), axis=-1)
        #    print(mask.shape[-1])
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
            #print(np.shape(image))
            
            #image=np.transpose(image,axes=(2,1,0))
        image=image.transpose(2,0,1)
            #mask=np.transpose(mask,axes=(2,1,0))
        
        return (tf.convert_to_tensor(image, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32))
        
    def __len__(self):
        return len(self.ids)
    
    
class My_Dataloder(keras.utils.Sequence):#utils.
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=2, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            #print(self.dataset[j].shape)
            data.append(self.dataset[j])
        
        # transpose list of lists
        #batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        batch = [tf.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def getitem(self, i):
        return self.__getitem__(i)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   

# Lets look at data we have
#dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])