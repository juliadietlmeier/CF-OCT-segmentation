# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:15:37 2019

@author: dietlmj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:46:25 2019

@author: dietlmj
"""

# Drosophila membrane ISBI challenge data
# https://github.com/zhixuhao/unet/blob/master/data.py
# -*- coding: utf-8 -*-

from numpy.random import seed
seed(1)
import tensorflow as tf
#from tensorflow import keras
#from OHSU_cardiac_data import *
#from my_dgs import *
from dataloader_Segformer import *
from seg_acc_metrics import seg_acc
import tensorflow.keras as keras
from keras.optimizers import Adam
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
import pandas as pd
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
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
#from keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
#from keras import backend as keras
#from keras.preprocessing.image import ImageDataGenerator
#import imageio
import tensorflow.keras.backend as K
from models_4paper import unet1024_GN, UEfficientNet, build_vgg16_unet

import os
#os.environ['TF_USE_LEGACY_KERAS'] = "1"

def dice_coef(y_true, y_pred, smooth=1e-10):#1e-7
    #y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,0:])#was 1 [Ellipsis,1:])
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

x_train_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/new_all_images_TRAIN/'
y_train_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/new_all_masks_TRAIN/'


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

image_size = 256
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    #input_mask -= 1
    return input_image, input_mask


class MyFunctor():
    def __init__(self, input_image, input_mask):
        self.input_image = input_image
        self.input_mask = input_mask
    def __call__(self, x):
        input_image = tf.image.convert_image_dtype(input_image, tf.float32)
        input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
        input_mask -= 1
        return input_image, input_mask

# Dataset for train images
train_dataset = My_Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=['fibrosis', 'ci_track', 'st_free_space', 'background'])
    #preprocessing = MyFunctor()
#)



train_dataloader = My_Dataloder(train_dataset, batch_size=2, shuffle=True)

# define callbacks for learning rate scheduling and best checkpoints saving
#callbacks = [
#    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
#    keras.callbacks.ReduceLROnPlateau(),
#]
model_checkpoint = ModelCheckpoint('OCT_4paper_SegFormer.keras', monitor='loss',verbose=1, save_best_only=True)

#model = unet1024_GN()
#------------------------------------------------------------------------------
#img_size = 256
#model = UEfficientNet(input_shape=(img_size,img_size,3),dropout_rate=0.25)
#------------------------------------------------------------------------------
input_shape = (256, 256, 3)
#model = build_vgg16_unet(input_shape)

from transformers import TFSegformerForSemanticSegmentation

model_checkpoint = "nvidia/mit-b0"#"nvidia/mit-b0"
id2label = {0: "background", 1: "ST_free_space", 2: "CI_track", 3: "Fibrosis"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)

model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

#segformer_model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")
segformer_model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

new_config = segformer_model.config
new_config.num_labels=num_labels
new_config.id2label = id2label
new_config.label2id=label2id
new_config.ignore_mismatched_sizes=True

# Instantiate new (randomly initialized) model
new_model = TFSegformerForSemanticSegmentation(new_config)

#------------------------------------------------------------------------------

lr = 0.00006
opt = Adam(learning_rate=lr)

metrics = [dice_coef, 'accuracy']
#new_model.compile(loss=dice_loss, optimizer='adam', metrics=metrics)#dice_loss
new_model.compile(optimizer='adam')
# train model


"==="
data_generator = My_Dataloder(train_dataset, batch_size=2, shuffle=True)

def gen_data_generator():
    for i in range(len(data_generator)):#(data_generator.len):
        yield data_generator.getitem(i)    #edited regaring to @Inigo-13 comment

#data_dataset =  tf.data.Dataset.from_generator(gen_data_generator, output_signature=(
#        tf.TensorSpec(shape=[None, 3, 512,512], dtype=tf.float32),
#        tf.TensorSpec(shape=[None, 512,512], dtype=tf.float32)))  #according to tf.data.Dataset.from_generator documentation we have to specify output_signature


data_dataset = tf.data.Dataset.from_generator(gen_data_generator,output_types=(tf.float32, tf.float32))

input_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/train_images/'
target_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/train_masks/'


input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

#dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))

data_dataset = get_dataset(
    2,
    (512,512),
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
)

#data_dataset = tf.data.Dataset.from_generator(
#        gen_data_generator, (tf.float32, tf.float32), (tf.TensorShape([None,3,512,512]), tf.TensorShape([None])))

model.compile(optimizer='adam')
#model.fit(x=data_dataset)

history = model.fit(#_generator
    x=data_dataset, 
    steps_per_epoch=len(data_dataset), 
    epochs=200)

#print(history.history.keys())
#plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')

#plt.figure()
#plt.plot(history.history['dice_coef'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model dice')
#plt.ylabel('dice')
#plt.xlabel('epoch')




"=============================================================================="
"==== PREDICTIONS ============================================================="

#testGene = testGenerator('/home/daa/OHSU/Cochlear_project/Cochlear_project/Cochlear_May2023/UNET_Annotations/image/', num_image=number_test_patches)

"=== create test data ========================================================="
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
#model.load_weights("OCT_4paper_model.keras")

#results=model.predict(trainX)
#results = model.predict_generator(testGene, number_test_patches, verbose=1)#72 for overlap=20 with size=256

#from OHSU_cardiac_data import *
#saveResult("/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_results/", results)

mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])

def normalize(input_image):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    #input_mask -= 1
    return input_image

"=============================================================================="
from skimage import color
from metrics_multiclass import *

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/corrected/predictions/SegFormer/'
dst_res_overlayed='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/overlayed_predictions/'

x_test_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/test_images/'
y_test_dir='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/New_multiclass_annotations/4paper/dataset/correct_dataset/split/test_masks/'

path, dirs, files = (next(os.walk(x_test_dir)))
path_y, dirs_y, files_y = (next(os.walk(y_test_dir)))

k=0
from skimage.transform import resize
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
    img=cv2.resize(img,(512,512))
    img_original=img
#----rescale ------------------------------------------------------------------    
    img = normalize(img/255)#img/255 # for UNET and VGG16
    #img = img # for Efficient Net no rescaling is needed
#------------------------------------------------------------------------------
    img = np.transpose(img,(2,0,1))
    img_original = np.transpose(img_original,(2,0,1))
    #img=np.expand_dims(img[:,:,0],-1)
    img=np.expand_dims(img,0)
    img_original=np.expand_dims(img_original,0)
    
    print('shape img = ', np.shape(img))
    results=model.predict(img).logits
    
    #pre=np.argmax(results[0,:,:,:], -1)
    pre = tf.math.argmax(results, axis=1)
    #pre = tf.expand_dims(pre, -1)
    pre = np.uint8(pre[0])

    
    
    pre = np.asarray(pre)
    #pre = np.squeeze(pre,axis=2)
    pre = resize(pre, (512,512))
    pre = (pre-np.min(pre))/(np.max(pre)-np.min(pre))
    pre=np.uint8(pre*3)
    
    true=cv2.imread(path_y+i,0)
    true=cv2.resize(true,(512,512))
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
    
    image=np.squeeze(img_original[:,0,:,:],0)
    
    #pre = (pre+np.mean(mean))*np.mean(std)
    
    res=color.label2rgb(pre,image,colors=[(255,0,0),(0,0,255), (0,255,0)],alpha=0.1, bg_label=0, bg_color=None)*255
    cv2.imwrite(dst_res+i.split('.')[0]+'_predict.png', res)
    k=k+1

seg_accuracy=np.mean(np.asarray(seg_acc_arr),axis=0)
seg_precision=np.mean(np.asarray(precision_arr),axis=0)
seg_recall=np.mean(np.asarray(recall_arr),axis=0)
#seg_Fscore=np.mean(np.asarray(Fscore_arr))
seg_Jaccard=np.mean(np.asarray(Jaccard_arr),axis=0)
seg_dice=np.mean(np.asarray(dice_arr),axis=0)


"=============================================================================="

m



"=== PREDICT on whole 300 to 550 slices = 250 slices and postprocess to get the Fibrosis content"
from skimage import data, filters, measure, morphology

dst_res='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_results_OCTV7L_ALL/'
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
    img=np.expand_dims(img1[:,:,0],-1)
    img=np.expand_dims(img,0)  # input to the model should be (1,width,height,numbre_channels)
    
    result=model.predict(img)
    pre=np.argmax(result,-1)
    pre=np.squeeze(pre,axis=0)
# overlay and save overlayed results
    res=color.label2rgb(pre,img1,colors=[(255,0,0),(0,0,255)],alpha=0.1, bg_label=0, bg_color=None)*255
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
    
    [r,c]=np.where(pre==2)# Fibrosis
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
plt.savefig('OCTV7L Fibrosis amount for each slice in visible spectrum')



"Overlay GT onto the Fibrosis plots============================================"

src_dir_img='/home/daa/Desktop/Cochlear_project/Cochlear_May2023/UNET_Annotations/November2023/OCTV7L/train_label_OCTV7L_04/'
#idx=np.arange(198,322,1)#300 to 550

idx=[166,171,176,186,196,201,206,211,
           216,221,226,231,236,241,246,251,
           256,261,266,271,276,281,286,323,
           389,462,483,512]

path, dirs, files = next(os.walk(src_dir_img))

# 0198_gt.png
Fibrosis_arr_GT=[]
Fibrosis_arr_2=[]
for k in idx:
    fname=str(k).zfill(4)+'_gt.png'
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
    [r,c]=np.where(pre==1)# ST Free Space
    mask_ST=np.zeros((1024,1024))
    mask_ST[r,c]=1
    labels_ST = measure.label(mask_ST)
    props_ST = measure.regionprops(labels_ST, img1)
    props = measure.regionprops_table(labels_ST, img1, properties=['area'])
    ST_area=np.sum(props['area'])
    
    #[r,c]=np.where(pre==2)# CI_track
    
    [r,c]=np.where(pre==2)# Fibrosis
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
plt.plot(np.arange(150,515,1),Fibrosis_arr[150:515],'-',label='AI-computed Fibrosis')
plt.plot(idx,Fibrosis_arr_GT,'.-r',label='Human-computed Fibrosis'),
plt.ylim([10,100])
plt.ylabel('Fibrosis in scala tympani (%)')
plt.xlabel('Volume depth')
plt.legend()
plt.title('OCTV7L AI-computed Fibrosis amount versus human annotations')
plt.savefig('OCTV7L AI-computed Fibrosis amount versus human annotations v2')


import csv  
slices=np.arange(150,515,1)
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
        'AI_Fibrosis': Fibrosis_arr[150:515],
        #'slice_number_Human':idx,
        #'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('Fibrosis_OCTV7L_AI.csv', index=False)
#------------------------------------------------------------------------------
Biodata = {'slice_number_human': idx,
        'Human_Fibrosis': Fibrosis_arr_GT
        }
df = pd.DataFrame(Biodata)
df.to_csv('Fibrosis_OCTV7L_human.csv', index=False)