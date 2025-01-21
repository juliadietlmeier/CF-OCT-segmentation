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
from tensorflow import keras

class Dataset:
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
        image = cv2.resize(image,(1024,1024))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#--- rescale ------------------------------------------------------------------        
        image = image/255# for UNET and VGG16 and MSTDeepLabV3+
        #image = image # for Efficient Net no rescaling is needed
#------------------------------------------------------------------------------ 
        #print('image size = ', np.shape(image))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask,(1024,1024))
        #print(np.unique(mask))
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
        
        mask = new_mask
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
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
class Dataset_2class:
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
    
    CLASSES = ['fibrosis_ci_track', 'st_free_space', 'background']
    
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
        self.class_values = [0,1,2] #[self.CLASSES.index(cls.lower()) for cls in classes]
        print(self.class_values)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image,(1024,1024))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#--- rescale ------------------------------------------------------------------        
        image = image/255# for UNET and VGG16 and MSTDeepLabV3+
        #image = image # for Efficient Net no rescaling is needed
#------------------------------------------------------------------------------ 
        #print('image size = ', np.shape(image))
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask,(1024,1024))
        #print(np.unique(mask))
        #mask = mask/255
        #print('mask shape = ', np.shape(mask))
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #print(self.class_values)
        #mask = np.stack(masks, axis=-1).astype('float')
        
        
        new_mask = np.zeros(mask.shape + (3,))
        #---- one-hot encoding ------------------------------------------------
        new_mask[mask == 0.,   0] = 1
        new_mask[mask == 1.,   1] = 1
        new_mask[mask == 2.,   2] = 1
        #new_mask[mask == 3.,   3] = 1
        
        mask = new_mask
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
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

    
class Dataloder(keras.utils.Sequence):
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
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   

# Lets look at data we have
#dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])