# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:15:37 2025

@author: Julia Dietlmeier <julia.dietlmeier@insight-centre.org>
"""

from numpy.random import seed
seed(1)
import tensorflow as tf
from tensorflow import keras
#from OHSU_cardiac_data import *
#from my_dgs import *
from dataloader import *
from seg_acc_metrics import seg_acc

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
keras.utils.set_random_seed(812)

# This will make TensorFlow ops as deterministic as possible, but it will
# affect the overall performance, so it's not enabled by default.
# `enable_op_determinism()` is introduced in TensorFlow 2.9.
#tf.config.experimental.enable_op_determinism()

#from tensorflow import set_random_seed
#set_random_seed(2)

#from OHSU_seg_acc import OHSU_seg_acc
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
#import keras.backend as K
import tensorflow.keras.backend as K

from models_4paper import unet1024_GN, UEfficientNet, build_vgg16_unet
from deeplabv3_mobilenetv2 import DeeplabV3Plus

def dice_coef(y_true, y_pred, smooth=1e-10):#1e-7
    #y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,0:])#was 1 [Ellipsis,1:])
    #y_true_f = K.Flatten(y_true[...,0:])
    #y_pred_f = K.Flatten(y_pred[...,0:])#was 1 maybe because background was not included?
    
    y_true_f = Flatten()(y_true[...,0:])
    y_pred_f = Flatten()(y_pred[...,0:])#was 1 maybe because background was not included?
    
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

"=============================================================================="
data_gen_args = dict(rotation_range=    0.0,
                    width_shift_range=  0.0, # 0.3
                    height_shift_range= 0.0, # 0.3
                    shear_range=        0.0,
                    zoom_range=         0.0,
                    horizontal_flip=    False,
                    vertical_flip=      False,
                    fill_mode='nearest')

x_train_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/train_images/'
y_train_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/train_masks/'

# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=['fibrosis', 'ci_track', 'st_free_space', 'background']
)

train_dataloader = Dataloder(train_dataset, batch_size=2, shuffle=True)

# define callbacks for learning rate scheduling and best checkpoints saving
#callbacks = [
#    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
#    keras.callbacks.ReduceLROnPlateau(),
#]
#model_checkpoint = ModelCheckpoint('OCT_4paper_model.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint = ModelCheckpoint('OCT_4paper_model_OCTUNet.keras', monitor='loss',verbose=1, save_best_only=True)
#model = unet1024_GN()# make sure to set ng=2 in models_4paper.py 
#------------------------------------------------------------------------------
#img_size = 256
#model = UEfficientNet(input_shape=(img_size,img_size,3),dropout_rate=0.25)
#------------------------------------------------------------------------------
#input_shape = (256, 256, 3)
#model = build_vgg16_unet(input_shape)
model = DeeplabV3Plus(256, num_classes=4)
#------------------------------------------------------------------------------
m
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
metrics = [dice_coef, 'accuracy']

#loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)# for MSTDeepLabV3Plus
loss=dice_loss
model.compile(loss=loss, optimizer=opt, metrics=metrics)

#------------------------------------------------------------------------------

# train model
history = model.fit(#fit_generator
    train_dataloader, 
    steps_per_epoch=len(train_dataloader)//(2), 
    epochs=100, 
    callbacks=[model_checkpoint]
)

#print(history.history.keys())
plt.figure()
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')

plt.figure()
plt.plot(history.history['dice_coef'])
#plt.plot(history.history['val_accuracy'])
plt.title('model dice')
plt.ylabel('dice')
plt.xlabel('epoch')

plt.figure()
plt.plot(history.history['loss'])
#plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
"=============================================================================="
"==== PREDICTIONS ============================================================="

#testGene = testGenerator('/home/daa/OHSU/Cochlear_project/Cochlear_project/Cochlear_May2023/UNET_Annotations/image/', num_image=number_test_patches)

#"=== create test data ========================================================="
#import glob

#def load_test_data():

#  TEST_X = sorted(glob.glob('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV7L_04/OCTV7L_04/*.jpg'))
#  test_x = TEST_X

#  return test_x

#test_x = load_test_data() 
#import cv2
#testX=[]
#for i in range(len(test_x)):
#    img=skimage.io.imread(test_x[i])
#    img=cv2.resize(img,(1024,1024))
#    testX.append(np.expand_dims(img[:,:,0],-1))

#testX=np.asarray(testX)

"=============================================================================="
#model = unet1024_GN()
#model = UEfficientNet(input_shape=(img_size,img_size,3),dropout_rate=0.25)
#model = build_vgg16_unet(input_shape)
#model = DeeplabV3Plus(256, num_classes=4)
#model.load_weights("OCT_4paper_model_OCTUNet.keras")

#results=model.predict(trainX)
#results = model.predict_generator(testGene, number_test_patches, verbose=1)#72 for overlap=20 with size=256

#from OHSU_cardiac_data import *
#saveResult("/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_results/", results)



"=============================================================================="
from skimage import color
from metrics_multiclass import *
import os

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/predictions/OCT_UNET/'
#dst_res_overlayed='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/overlayed_predictions/UEfficientNET/'
if not os.path.exists(dst_res):
    os.makedirs(dst_res)


x_test_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/test_images/'
y_test_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/test_masks/'

path, dirs, files = (next(os.walk(x_test_dir)))
path_y, dirs_y, files_y = (next(os.walk(y_test_dir)))

k=0

seg_acc_arr=[]
precision_arr=[]
recall_arr=[]
Fscore_arr=[]
Jaccard_arr=[]
dice_arr=[]

for i in sorted(files):
    #index=int(i.split('.')[0])
    img=cv2.imread(path+i)
    #img=[img,img,img]
    img=cv2.resize(img,(1024,1024))
#----rescale ------------------------------------------------------------------    
    img = img/255 # for UNET and VGG16 and MSTDeepLabV3+
    #img = img # for Efficient Net no rescaling is needed
#------------------------------------------------------------------------------
    #img=np.expand_dims(img[:,:,0],-1)
    img=np.expand_dims(img,0)
    
    results=model.predict(img)
    pre=np.argmax(results[0,:,:,:],-1)
    
    true=cv2.imread(path_y+i,0)
    true=cv2.resize(true,(1024,1024))
    #true = true/255
    print(np.shape(true))
    
    FP, FN, TP, TN = numeric_score(pre, true)
    seg_acc = accuracy_score(FP, FN, TP, TN)
    precision = precision_score(FP, FN, TP, TN)
    recall = recall_score(FP, FN, TP, TN)
    #Fscore=
    Jaccard = jaccard_score(pre,true)
    DSC=my_dice_score(pre,true)
    
    seg_acc_arr.append(seg_acc)
    precision_arr.append(precision)
    recall_arr.append(recall)
    #Fscore_arr.append(Fscore)
    Jaccard_arr.append(Jaccard)
    dice_arr.append(DSC)
    
    image=np.squeeze(img[:,:,:,0],0)
    res=color.label2rgb(pre,image,colors=[(255,0,0),(0,0,255), (0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+i.split('.')[0]+'_predict.png', res)
    k=k+1

seg_accuracy=np.mean(np.asarray(seg_acc_arr),axis=0)
seg_precision=np.mean(np.asarray(precision_arr),axis=0)
seg_recall=np.mean(np.asarray(recall_arr),axis=0)
#seg_Fscore=np.mean(np.asarray(Fscore_arr))
seg_Jaccard=np.mean(np.asarray(Jaccard_arr),axis=0)
seg_dice=np.mean(np.asarray(dice_arr),axis=0)

# print average statistics
print('ave_seg_dice=',np.mean(seg_dice))
print('ave_seg_accuracy=',np.mean(seg_accuracy))
print('ave_seg_precision=',np.mean(seg_precision))
print('ave_seg_recall=',np.mean(seg_recall))
print('ave_seg_Jaccard=',np.mean(seg_Jaccard))


"=============================================================================="
"==== OCTV7L =================================================================="

import glob
import skimage
from skimage import data, filters, measure, morphology

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/UNET_results_OCTV7L_ALL/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV7L_04/OCTV7L_04/'
idx=np.arange(0,1024,1)#300 to 550
path, dirs, files = next(os.walk(src_dir_img))
test_files = sorted(glob.glob('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV7L_07/OCTV7L_04/*.jpg'))
# img format OCTV1L_0001_Mode3D_page_sliceidx.jpg
Fibrosis_arr=[]
Fibrosis_arr_2=[]
for k in idx:
    fname='OCTV7L_04_'+str(k).zfill(4)+'.jpg'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    #img=np.expand_dims(img1[:,:,0],-1)
    img=np.expand_dims(img1,0)  # input to the model should be (1,width,height,numbre_channels)
    
    result=model.predict(img)
    pre=np.argmax(result,-1)
    pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,0,255),(0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    
    [r,c]=np.where(pre==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr=np.asarray(Fibrosis_arr)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)
plt.figure()
plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV7L Fibrosis amount for each slice in visible spectrum')

plt.figure()
plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV7L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(150,515,1),Fibrosis_arr[150:515],'-'),plt.title('OCTV7L Fibrosis amount for each slice in visible spectrum')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV7L Fibrosis amount for each slice in visible spectrum')



"Overlay GT onto the Fibrosis plots============================================"

#src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_Annotations/November2023/OCTV7L/train_label_OCTV7L_04/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/sorted_raw_volumes/masks/OCTV7L/'
#idx=np.arange(198,322,1)#300 to 550

#idx=[166,171,176,186,196,201,206,211,
#           216,221,226,231,236,241,246,251,
#           256,261,266,271,276,281,286,323,
#           389,462,483,512]

idx=[196,201,206,211,216,221,226,231,
     236,241,246,251,256,261,266,271,
     276,281,286,290,294,323]

path, dirs, files = next(os.walk(src_dir_img))

# 0198_gt.png
Fibrosis_arr_GT=[]
Fibrosis_arr_2=[]
for k in idx:
    #fname=str(k).zfill(4)+'_gt.png'
    fname='OCTV7L_04_'+str(k).zfill(4)+'.png'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    
    #img=np.expand_dims(img1[:,:,0],-1)
    #img=np.expand_dims(img,0)  # input to the model should be (1,width,height,numbre_channels)
    
    #result=model.predict(img)
    #pre=np.argmax(result,-1)
    #pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    #res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,85,0),(0,0,255)],alpha=0.1, bg_label=0, bg_color=None)*255
    #cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    pre=img
    [r,c]=np.where(pre[:,:,0]==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre[:,:,0]==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr_GT.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr_GT=np.asarray(Fibrosis_arr_GT)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)

#plt.figure()
#plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV8L Fibrosis amount for each slice in visible spectrum')

#plt.figure()
#plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV8L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(190,350,1),Fibrosis_arr[190:350],'-',label='AI-computed Fibrosis')
plt.plot(idx,Fibrosis_arr_GT,'.-r',label='Human-computed Fibrosis'),
plt.ylim([0,100])
plt.ylabel('Fibrosis in scala tympani (%)')
plt.xlabel('Volume depth')
plt.legend()
plt.title('OCTV7L AI-computed Fibrosis amount versus human annotations')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV7L AI-computed Fibrosis amount versus human annotations')


import csv  
slices=np.arange(190,350,1)
#header = ['slice_number', 'AI_Fibrosis', 'Human_Fibrosis']
#data = [slices, Fibrosis_arr, Fibrosis_arr_GT]

#with open('Fibrosis_OCTV1L.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
    # write the header
#    writer.writerow(header)
    # write the data
#    writer.writerow(data)

import pandas as pd

# Create a sample dataframe
Biodata = {'slice_number_AI': slices,
        'AI_Fibrosis': Fibrosis_arr[190:350],
        #'slice_number_Human':idx,
        #'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV7L_AI.csv', index=False)
#------------------------------------------------------------------------------
Biodata = {'slice_number_human': idx,
        'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV7L_human.csv', index=False)

"=============================================================================="
"OCTV1L"

import glob
import skimage
from skimage import data, filters, measure, morphology

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/UNET_results_OCTV1L_ALL/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV1L_0001_Mode3D/'
idx=np.arange(0,1024,1)#300 to 550
path, dirs, files = next(os.walk(src_dir_img))
test_files = sorted(glob.glob('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV1L_0001_Mode3D/*.jpg'))
# img format OCTV1L_0001_Mode3D_page_sliceidx.jpg
Fibrosis_arr=[]
Fibrosis_arr_2=[]
for k in idx:
    #fname='OCTV7L_04_'+str(k).zfill(4)+'.jpg'
    fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.jpg'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    #img=np.expand_dims(img1[:,:,0],-1)
    img=np.expand_dims(img1,0)  # input to the model should be (1,width,height,numbre_channels)
    
    result=model.predict(img)
    pre=np.argmax(result,-1)
    pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,0,255), (0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    
    [r,c]=np.where(pre==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr=np.asarray(Fibrosis_arr)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)
plt.figure()
plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV1L Fibrosis amount for each slice in visible spectrum')

plt.figure()
plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV1L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(280,560,1),Fibrosis_arr[280:560],'-'),plt.title('OCTV1L Fibrosis amount for each slice in visible spectrum')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV1L Fibrosis amount for each slice in visible spectrum')



"Overlay GT onto the Fibrosis plots============================================"

#src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_Annotations/November2023/OCTV7L/train_label_OCTV7L_04/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/sorted_raw_volumes/masks/OCTV1L/'
#idx=np.arange(198,322,1)#300 to 550

#idx=[166,171,176,186,196,201,206,211,
#           216,221,226,231,236,241,246,251,
#           256,261,266,271,276,281,286,323,
#           389,462,483,512]

idx=[289,292,296,297,300,304,309,313,
     317,320,325,329,332,336,339,341,
     345,349,350,354,357,360,362,366,
     370,374,377,379,382,386,390,394,
     397,400,404,408,411,413,417,420,
     424,428,431,435,439,441,445,449,
     452,455,459,462,466,468,471,474,
     478,480,484,487,490,494,498,501,
     505,507,510,514,518,520,524,527,
     531,535,539,542,546]

path, dirs, files = next(os.walk(src_dir_img))

# 0198_gt.png
Fibrosis_arr_GT=[]
Fibrosis_arr_2=[]
for k in idx:
    #fname=str(k).zfill(4)+'_gt.png'
    #fname='OCTV7L_04_'+str(k).zfill(4)+'.png'
    fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.png'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    
    #img=np.expand_dims(img1[:,:,0],-1)
    #img=np.expand_dims(img,0)  # input to the model should be (1,width,height,numbre_channels)
    
    #result=model.predict(img)
    #pre=np.argmax(result,-1)
    #pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    #res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,85,0),(0,0,255)],alpha=0.1, bg_label=0, bg_color=None)*255
    #cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    pre=img
    [r,c]=np.where(pre[:,:,0]==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre[:,:,0]==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr_GT.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr_GT=np.asarray(Fibrosis_arr_GT)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)

#plt.figure()
#plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV8L Fibrosis amount for each slice in visible spectrum')

#plt.figure()
#plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV8L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(280,560,1),Fibrosis_arr[280:560],'-',label='AI-computed Fibrosis')
plt.plot(idx,Fibrosis_arr_GT,'.-r',label='Human-computed Fibrosis'),
plt.ylim([0,100])
plt.ylabel('Fibrosis in scala tympani (%)')
plt.xlabel('Volume depth')
plt.legend()
plt.title('OCTV1L AI-computed Fibrosis amount versus human annotations')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV1L AI-computed Fibrosis amount versus human annotations')


import csv  
slices=np.arange(280,560,1)
#header = ['slice_number', 'AI_Fibrosis', 'Human_Fibrosis']
#data = [slices, Fibrosis_arr, Fibrosis_arr_GT]

#with open('Fibrosis_OCTV1L.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
    # write the header
#    writer.writerow(header)
    # write the data
#    writer.writerow(data)

import pandas as pd

# Create a sample dataframe
Biodata = {'slice_number_AI': slices,
        'AI_Fibrosis': Fibrosis_arr[280:560],
        #'slice_number_Human':idx,
        #'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV1L_AI.csv', index=False)
#------------------------------------------------------------------------------
Biodata = {'slice_number_human': idx,
        'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV1L_human.csv', index=False)

"=============================================================================="
"OCTV9L"

import glob
import skimage
from skimage import data, filters, measure, morphology

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/UNET_results_OCTV9L_ALL/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV9L_03/OCTV9L_03/'
idx=np.arange(0,1024,1)#300 to 550
path, dirs, files = next(os.walk(src_dir_img))
test_files = sorted(glob.glob('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV9L_03/OCTV9L_03/*.jpg'))
# img format OCTV1L_0001_Mode3D_page_sliceidx.jpg
Fibrosis_arr=[]
Fibrosis_arr_2=[]
for k in idx:
    fname='OCTV9L_03_'+str(k).zfill(4)+'.jpg'
    #fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.jpg'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    #img=np.expand_dims(img1[:,:,0],-1)
    img=np.expand_dims(img1,0)  # input to the model should be (1,width,height,numbre_channels)
    
    result=model.predict(img)
    pre=np.argmax(result,-1)
    pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,0,255), (0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    
    [r,c]=np.where(pre==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr=np.asarray(Fibrosis_arr)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)
plt.figure()
plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV9L Fibrosis amount for each slice in visible spectrum')

plt.figure()
plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV9L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(300,415,1),Fibrosis_arr[300:415],'-'),plt.title('OCTV9L Fibrosis amount for each slice in visible spectrum')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV9L Fibrosis amount for each slice in visible spectrum')



"Overlay GT onto the Fibrosis plots============================================"

#src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_Annotations/November2023/OCTV7L/train_label_OCTV7L_04/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/sorted_raw_volumes/masks/OCTV9L/'
#idx=np.arange(198,322,1)#300 to 550

#idx=[166,171,176,186,196,201,206,211,
#           216,221,226,231,236,241,246,251,
#           256,261,266,271,276,281,286,323,
#           389,462,483,512]

idx=[316,320,324,328,332,336,340,344,
     348,352,356,360,364,368,372,376,
     380,383,384,388,392,396,400,404]

path, dirs, files = next(os.walk(src_dir_img))

# 0198_gt.png
Fibrosis_arr_GT=[]
Fibrosis_arr_2=[]
for k in idx:
    #fname=str(k).zfill(4)+'_gt.png'
    fname='OCTV9L_03_'+str(k).zfill(4)+'.png'
    #fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.png'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    
    #img=np.expand_dims(img1[:,:,0],-1)
    #img=np.expand_dims(img,0)  # input to the model should be (1,width,height,numbre_channels)
    
    #result=model.predict(img)
    #pre=np.argmax(result,-1)
    #pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    #res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,85,0),(0,0,255)],alpha=0.1, bg_label=0, bg_color=None)*255
    #cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    pre=img
    [r,c]=np.where(pre[:,:,0]==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre[:,:,0]==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr_GT.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr_GT=np.asarray(Fibrosis_arr_GT)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)

#plt.figure()
#plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV8L Fibrosis amount for each slice in visible spectrum')

#plt.figure()
#plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV8L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(300,415,1),Fibrosis_arr[300:415],'-',label='AI-computed Fibrosis')
plt.plot(idx,Fibrosis_arr_GT,'.-r',label='Human-computed Fibrosis'),
plt.ylim([0,100])
plt.ylabel('Fibrosis in scala tympani (%)')
plt.xlabel('Volume depth')
plt.legend()
plt.title('OCTV9L AI-computed Fibrosis amount versus human annotations')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV9L AI-computed Fibrosis amount versus human annotations')


import csv  
slices=np.arange(300,415,1)
#header = ['slice_number', 'AI_Fibrosis', 'Human_Fibrosis']
#data = [slices, Fibrosis_arr, Fibrosis_arr_GT]

#with open('Fibrosis_OCTV1L.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
    # write the header
#    writer.writerow(header)
    # write the data
#    writer.writerow(data)

import pandas as pd

# Create a sample dataframe
Biodata = {'slice_number_AI': slices,
        'AI_Fibrosis': Fibrosis_arr[300:415],
        #'slice_number_Human':idx,
        #'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV9L_AI.csv', index=False)
#------------------------------------------------------------------------------
Biodata = {'slice_number_human': idx,
        'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV9L_human.csv', index=False)

"=============================================================================="
"OCTV10L"

import glob
import skimage
from skimage import data, filters, measure, morphology

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/UNET_results_OCTV10L_ALL/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV10L_02/'
idx=np.arange(0,1024,1)#300 to 550
path, dirs, files = next(os.walk(src_dir_img))
test_files = sorted(glob.glob('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV10L_02/*.jpg'))
# img format OCTV1L_0001_Mode3D_page_sliceidx.jpg
Fibrosis_arr=[]
Fibrosis_arr_2=[]
for k in idx:
    fname='OCTV10L_2_'+str(k).zfill(4)+'.jpg'
    #fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.jpg'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    #img=np.expand_dims(img1[:,:,0],-1)
    img=np.expand_dims(img1,0)  # input to the model should be (1,width,height,numbre_channels)
    
    result=model.predict(img)
    pre=np.argmax(result,-1)
    pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,0,255), (0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    
    [r,c]=np.where(pre==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr=np.asarray(Fibrosis_arr)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)
plt.figure()
plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV10L Fibrosis amount for each slice in visible spectrum')

plt.figure()
plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV10L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(200,345,1),Fibrosis_arr[200:345],'-'),plt.title('OCTV10L Fibrosis amount for each slice in visible spectrum')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV10L Fibrosis amount for each slice in visible spectrum')



"Overlay GT onto the Fibrosis plots============================================"

#src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_Annotations/November2023/OCTV7L/train_label_OCTV7L_04/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/sorted_raw_volumes/masks/OCTV10L/'
#idx=np.arange(198,322,1)#300 to 550

#idx=[166,171,176,186,196,201,206,211,
#           216,221,226,231,236,241,246,251,
#           256,261,266,271,276,281,286,323,
#           389,462,483,512]

idx=[212,216,220,224,228,232,236,240,
     244,248,252,256,264,268,276,280,
     284,288,292,296,300,304,308,312,
     316,320,324,328,332,336]

path, dirs, files = next(os.walk(src_dir_img))

# 0198_gt.png
Fibrosis_arr_GT=[]
Fibrosis_arr_2=[]
for k in idx:
    #fname=str(k).zfill(4)+'_gt.png'
    fname='OCTV10L_2_'+str(k).zfill(4)+'.png'
    #fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.png'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    
    #img=np.expand_dims(img1[:,:,0],-1)
    #img=np.expand_dims(img,0)  # input to the model should be (1,width,height,numbre_channels)
    
    #result=model.predict(img)
    #pre=np.argmax(result,-1)
    #pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    #res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,85,0),(0,0,255)],alpha=0.1, bg_label=0, bg_color=None)*255
    #cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    pre=img
    [r,c]=np.where(pre[:,:,0]==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre[:,:,0]==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr_GT.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr_GT=np.asarray(Fibrosis_arr_GT)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)

#plt.figure()
#plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV8L Fibrosis amount for each slice in visible spectrum')

#plt.figure()
#plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV8L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(200,345,1),Fibrosis_arr[200:345],'-',label='AI-computed Fibrosis')
plt.plot(idx,Fibrosis_arr_GT,'.-r',label='Human-computed Fibrosis'),
plt.ylim([0,100])
plt.ylabel('Fibrosis in scala tympani (%)')
plt.xlabel('Volume depth')
plt.legend()
plt.title('OCTV10L AI-computed Fibrosis amount versus human annotations')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV10L AI-computed Fibrosis amount versus human annotations')


import csv  
slices=np.arange(200,345,1)
#header = ['slice_number', 'AI_Fibrosis', 'Human_Fibrosis']
#data = [slices, Fibrosis_arr, Fibrosis_arr_GT]

#with open('Fibrosis_OCTV1L.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
    # write the header
#    writer.writerow(header)
    # write the data
#    writer.writerow(data)

import pandas as pd

# Create a sample dataframe
Biodata = {'slice_number_AI': slices,
        'AI_Fibrosis': Fibrosis_arr[200:345],
        #'slice_number_Human':idx,
        #'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV10L_AI.csv', index=False)
#------------------------------------------------------------------------------
Biodata = {'slice_number_human': idx,
        'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV10L_human.csv', index=False)

"=============================================================================="
"OCTV11L"

import glob
import skimage
from skimage import data, filters, measure, morphology

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/UNET_results_OCTV11L_ALL/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV11L_01/'
idx=np.arange(0,1024,1)#300 to 550
path, dirs, files = next(os.walk(src_dir_img))
test_files = sorted(glob.glob('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/images_jpgs/jpgs/OCTV11L_01/*.jpg'))
# img format OCTV1L_0001_Mode3D_page_sliceidx.jpg
Fibrosis_arr=[]
Fibrosis_arr_2=[]
for k in idx:
    fname='OCTV11L_01_'+str(k).zfill(4)+'.jpg'
    #fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.jpg'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    #img=np.expand_dims(img1[:,:,0],-1)
    img=np.expand_dims(img1,0)  # input to the model should be (1,width,height,numbre_channels)
    
    result=model.predict(img)
    pre=np.argmax(result,-1)
    pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,0,255), (0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    
    [r,c]=np.where(pre==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr=np.asarray(Fibrosis_arr)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)
plt.figure()
plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV11L Fibrosis amount for each slice in visible spectrum')

plt.figure()
plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV11L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(320,430,1),Fibrosis_arr[320:430],'-'),plt.title('OCTV11L Fibrosis amount for each slice in visible spectrum')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV11L Fibrosis amount for each slice in visible spectrum')



"Overlay GT onto the Fibrosis plots============================================"

#src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_Annotations/November2023/OCTV7L/train_label_OCTV7L_04/'
src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/sorted_raw_volumes/masks/OCTV11L/'
#idx=np.arange(198,322,1)#300 to 550

#idx=[166,171,176,186,196,201,206,211,
#           216,221,226,231,236,241,246,251,
#           256,261,266,271,276,281,286,323,
#           389,462,483,512]

idx=[338,346,350,354,358,362,366,370,
     374,378,382,386,390,394,398,402,
     406,410,414,418]

path, dirs, files = next(os.walk(src_dir_img))

# 0198_gt.png
Fibrosis_arr_GT=[]
Fibrosis_arr_2=[]
for k in idx:
    #fname=str(k).zfill(4)+'_gt.png'
    fname='OCTV11L_01_'+str(k).zfill(4)+'.png'
    #fname = 'OCTV1L_0001_Mode3D_page_'+str(k)+'.png'
    img=skimage.io.imread(path+fname)
    img1=cv2.resize(img,(1024,1024))
    
    #img=np.expand_dims(img1[:,:,0],-1)
    #img=np.expand_dims(img,0)  # input to the model should be (1,width,height,numbre_channels)
    
    #result=model.predict(img)
    #pre=np.argmax(result,-1)
    #pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    #res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,85,0),(0,0,255)],alpha=0.1, bg_label=0, bg_color=None)*255
    #cv2.imwrite(dst_res+str(k)+'_predict.png', res)
# calculate CI/track, Fibrosis and ST Free Space areas
    pre=img
    [r,c]=np.where(pre[:,:,0]==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre[:,:,0]==3)# Fibrosis
    mask_F=np.zeros((1024,1024))
    mask_F[r,c]=1
    labels_F = measure.label(mask_F)
    props_F = measure.regionprops(labels_F, img1)
    props_F = measure.regionprops_table(labels_F, img1, properties=['area'])
    Fibrosis_area=np.sum(props_F['area'])
    
    
    Amount_Fibrosis=(Fibrosis_area*100)/(ST_area+Fibrosis_area)
    Amount_Fibrosis_2=(Fibrosis_area*100*np.max((Fibrosis_area,ST_area)))/(ST_area+Fibrosis_area)
    Fibrosis_arr_GT.append(Amount_Fibrosis)
    Fibrosis_arr_2.append(Amount_Fibrosis_2)
    
    print('page # = ',k)

Fibrosis_arr_GT=np.asarray(Fibrosis_arr_GT)
Fibrosis_arr_2=np.asarray(Fibrosis_arr_2)

#plt.figure()
#plt.plot(idx,Fibrosis_arr,'-'),plt.title('OCTV8L Fibrosis amount for each slice in visible spectrum')

#plt.figure()
#plt.plot(idx,Fibrosis_arr_2,'-'),plt.title('OCTV8L Normalized Fibrosis amount')
    
plt.figure()
plt.plot(np.arange(320,430,1),Fibrosis_arr[320:430],'-',label='AI-computed Fibrosis')
plt.plot(idx,Fibrosis_arr_GT,'.-r',label='Human-computed Fibrosis'),
plt.ylim([0,100])
plt.ylabel('Fibrosis in scala tympani (%)')
plt.xlabel('Volume depth')
plt.legend()
plt.title('OCTV11L AI-computed Fibrosis amount versus human annotations')
plt.savefig('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/OCTV11L AI-computed Fibrosis amount versus human annotations')


import csv  
slices=np.arange(320,430,1)
#header = ['slice_number', 'AI_Fibrosis', 'Human_Fibrosis']
#data = [slices, Fibrosis_arr, Fibrosis_arr_GT]

#with open('Fibrosis_OCTV1L.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
    # write the header
#    writer.writerow(header)
    # write the data
#    writer.writerow(data)

import pandas as pd

# Create a sample dataframe
Biodata = {'slice_number_AI': slices,
        'AI_Fibrosis': Fibrosis_arr[320:430],
        #'slice_number_Human':idx,
        #'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV11L_AI.csv', index=False)
#------------------------------------------------------------------------------
Biodata = {'slice_number_human': idx,
        'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/Fibrosis_csv_toGeorge/NEW_Fibrosis_OCTV11L_human.csv', index=False)

