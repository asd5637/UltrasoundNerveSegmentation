# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:18:49 2020
final code
including:
1. wiener
2. unet ensemble method
3. net-trim model
4. PCA remove false postive region
@author: lin
"""
import sys
import os
import glob
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import random as r
import skimage.transform as trans
import math

from keras.models import Model
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.decomposition import TruncatedSVD
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.abspath('/home/lin/Documents/final_code/final'))
from origin import create_origin
from GT import create_GT
from Filter import wiener_filter
from performance import *

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_image_dim_ordering("th")
#%%
"""
Parameters need to modify
1. folders: subject name for valiidation data 
2. mm: information for central distance, convert pixel to mm value (see data_v1.2.0.xlsx)
2. load_dir: directory for data, details in README file
3. save_dir: directory for saving results
"""
smooth = 1.

global image_rows
global image_cols
global load_dir
global save_dir
image_rows = 496
image_cols = 592
folders = ['kvgh15_v06', 'kvgh05_01_v05', 'kmuh06_v15', 'kmuh05_v08', 'kmuh02_v18', 'kvgh07_01_v07', 'kvgh04_v08', 'kmuh12_v02']
mm = [0.04, 0.05, 0.06, 0.09, 0.09, 0.05, 0.06, 0.08]
val_id=[]
pca_id=[]
num_of_val=8
num_of_aug=3
load_dir = '/home/lin/Documents/test/part*/data/origin/' 
save_dir = '/home/lin/Documents/test/results/'

#%%
"""
Read in data and split into training, validation data set
label: True with data augmentation 
output: npy file place in 'npy' folder
"""
def create_data(src, mask, label=False, gt='new', resize=(image_rows,image_cols)):
    # delete comments below for random validation subject
    """
    kvgh=0
    kmuh=0
    files = glob.glob(src + mask, recursive=True)
    
    np.random.seed(5)
    np.random.shuffle(files)
    
    for file_name in files:
        folder = file_name.split("origin/")[1]
        folder = folder.split("/")[0]
        if folder not in folders and 'kvgh' in folder and kvgh<num_of_val/2:
            #allfolder.append(folder)
            folders.append(folder)
            kvgh+=1

        elif folder not in folders and 'kmuh' in folder and kmuh<num_of_val/2:
            #allfolder.append(folder)
            folders.append(folder)
            kmuh+=1

        if folder in folders:
            total_val+=1
    total_val = int(total_val/3)
    """
    
    imgs_val=[]
    imgs_val_mask=[]
    imgs_train=[]
    imgs_train_mask=[]
    
    files = glob.glob(src + mask, recursive=True)    
    
    i = 0
    for file_name in files:
        if gt=='new':
            s1 = file_name.split('part')[0]
            gt_file_name = s1 + 'new_gt/'
        else:
            if 'part3' in file_name:
                continue
            else:
                s1 = file_name.split('part')[0]
                gt_file_name = s1 + 'kmuh_gt/'
        tmp = file_name.split("origin/")[1]
        folder = tmp.split("/")[0]
        number = tmp.split("/")[1]
        if folder in folders and '_cut' in file_name:
            file_name_src = file_name.replace('origin', 'filter')
            file_name_src = file_name_src.replace('_cut', '_f')
            file_name_mask = gt_file_name + folder + '/' + number
            file_name_mask = file_name_mask.replace('_cut', '_mask')
            
            img = io.imread(file_name_src, as_gray=True)
            img = trans.resize(img, resize, mode='constant', preserve_range=True)
            img = (img-img.mean()) / img.std()
            img = img.reshape((1,)+img.shape)
            img = np.array([img])
            
            imgs_val.append(img[0,:,:,:])
            val_id.append(folder)
        
            img_mask = io.imread(file_name_mask, as_gray=True)
            img_mask = trans.resize(img_mask, resize, mode = 'constant', preserve_range=True)
            img_mask /= 255
            img_mask = img_mask.reshape((1,)+img_mask.shape)
            img_mask = np.array([img_mask])
            
            imgs_val_mask.append(img_mask[0,:,:,:])


        elif folder not in folders and '_cut' in file_name:
            file_name_src = file_name.replace('origin', 'filter')
            file_name_src = file_name_src.replace('_cut', '_f')
            file_name_mask = gt_file_name + folder + '/' + number
            file_name_mask = file_name_mask.replace('_cut', '_mask')
            
            img = io.imread(file_name_src, as_gray=True)
            img = trans.resize(img, resize, mode='constant', preserve_range=True)
            img = (img-img.mean()) / img.std()
            img = img.reshape((1,)+img.shape)
            img = np.array([img])
            
            if label:
                img_aug = augmentation(img, num_of_aug)
                n = i % num_of_aug
                imgs_train.append(img_aug[0,:,:,:])
                imgs_train.append(img_aug[n,:,:,:])
            else:
                imgs_train.append(img[0,:,:,:])
            
            img_mask = io.imread(file_name_mask, as_gray=True)
            img_mask = trans.resize(img_mask, resize, mode='constant', preserve_range=True)
            img_mask /= 255
            img_mask = img_mask.reshape((1,)+img_mask.shape)
            img_mask = np.array([img_mask])
            
            if label:
                img_mask_aug = augmentation(img_mask, num_of_aug)
                n = i % num_of_aug
                imgs_train_mask.append(img_mask_aug[0,:,:,:])
                imgs_train_mask.append(img_mask_aug[n,:,:,:])
            else:
                imgs_train_mask.append(img_mask[0,:,:,:])
                
            i+=1
    if gt=='kmuh':
        npy_dir = save_dir + 'npy/kmuh/'
    elif gt=='new':
        npy_dir = save_dir + 'npy/new/'
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    np.save(npy_dir+'img_train.npy',np.array(imgs_train).astype('float32'))
    np.save(npy_dir+'img_train_mask.npy',np.array(imgs_train_mask).astype('float32'))
    np.save(npy_dir+'img_val.npy',np.array(imgs_val).astype('float32'))
    np.save(npy_dir+'img_val_mask.npy',np.array(imgs_val_mask).astype('float32'))
    
def augmentation(scans,n): 
    """
    parameters for augmentation 
    scans: source image
    n: produce n augmented images
    """
    datagen = ImageDataGenerator(
        featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=5,   
        horizontal_flip=True,   
        vertical_flip=False,  
        zoom_range=False)
    i=0
    scans_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=900): 
        scans_g=np.vstack([scans_g,batch])
        i += 1
        if i == n:
            break

    return scans_g

def dice_coef_np(y_true, y_pred):
    """
    evaluate prediction dice coefficient
    """
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def sensitivity(y_true, y_pred):
    """
    evaluate prediction sensitivity
    """
    y_pred[y_pred > 0.2] = 1    
    y_pred[y_pred != 1] = 0
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    TP = np.sum(np.logical_and(y_pred_f == 1, y_true_f == 1))
    FN = np.sum(np.logical_and(y_pred_f == 0, y_true_f == 1))
    if (TP+FN)==0:
        return 0
    else:
        return TP/(TP+FN)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#%%
"""
Define model

unet_trim: trimmed 5-layer u-net
unet_5l: original 5-layer u-net
unet-4l: 4-layer u-net
unet-3l: 3-layer u-net
"""
def unet_trim():
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)   

    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(116, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(116, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(233, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(233, (3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(426, (3, 3), activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(426, (3, 3), activation='relu', border_mode='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2,2), padding='same')(conv5), conv4], axis=1)
    conv6 = Conv2D(233, (3, 3), activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(233, (3, 3), activation='relu', border_mode='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(conv6), conv3], axis=1)
    conv7 = Conv2D(116, (3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(116, (3, 3), activation='relu', border_mode='same')(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def unet_5l():
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2,2), padding='same')(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def unet_4l():
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)  #same padding
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)   # stride = 2

    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv4)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(conv4), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def unet_3l():
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)  #same padding
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)   # stride = 2

    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv3), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

#%%
"""
Function for training model, called in the Train model section

x_train: images in training data
y_train: ground truth in training data
x_val: images in validation data
y_val: ground truth in validation data
layer: 3,4 or 5 layer u-net model
epoch: number of epoch for training
save_dir: directory for saving best weighting & training curve image
"""
def train_model(x_train, y_train, x_val, y_val, layer, epoch, save_dir):
    if layer==3:
        model = unet_3l()
    elif layer==4:
        model = unet_4l()
    else:
        model = unet_trim()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
        
    checkpoint = ModelCheckpoint(save_dir+str(layer)+'layer weight.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, batch_size=12, validation_split=0.0, validation_data=(x_val, y_val), epochs=epoch,callbacks = callbacks_list ,verbose=1, shuffle=True)
    
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_dir+str(layer)+'layer dice.png')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
#%%
"""
Called by Ensemble, PCA section

Return one subject data each time in validation dataset in purpose of PCA method
src: load_dir
allfolder: subject folder name in validation data
gt: gt_type
"""    
def val_folder(src, allfolder, mask, gt='new'):         
    src = src.replace('origin', 'filter')
    files = sorted(glob.glob(src + allfolder + mask))
    imgs_val=[]
    imgs_val_mask=[]
    
    for file_name in files:
        s1 = file_name.split('part')[0]
        s2 = file_name.split('filter')[1]
        if gt=='kmuh':
            gt_file_name = s1+'kmuh_gt'+s2
        else:
            gt_file_name = s1+'new_gt'+s2
        file_name_mask = gt_file_name.replace('_f', '_mask')
        
        img = io.imread(file_name, as_gray=True)
        img = trans.resize(img, (image_rows, image_cols), mode='constant', preserve_range=True)
        img = (img-img.mean()) / img.std()
        img = img.reshape((1,)+img.shape)            
        img = np.array([img])
        imgs_val.append(img[0,:,:,:])
        
        img_mask = io.imread(file_name_mask, as_gray=True)
        img_mask = trans.resize(img_mask, (image_rows, image_cols), mode='constant', preserve_range=True)
        img_mask /= 255
        img_mask = img_mask.reshape((1,)+img_mask.shape)
        img_mask = np.array([img_mask])
        imgs_val_mask.append(img_mask[0,:,:,:])
            
        
    imgs_val = np.array(imgs_val).astype('float32')
    imgs_val_mask = np.array(imgs_val_mask).astype('float32')        
    
    return imgs_val, imgs_val_mask
    
#%%
"""
Called by Ensemble, PCA section

pca(): fit PCA model
correct_mask(): project back to the image
parameter: n_component (lower to remove more region in images)
"""
def pca(pred):
    pred[pred > 0.2] = 1 
    pred[pred != 1] = 0
    Y = pred.squeeze()
    pca2 = TruncatedSVD(n_components=25).fit(Y.reshape(-1, image_rows*image_cols))
    
    return pca2
        
def correct_mask(Y_pred):
    Y_pred = (Y_pred > 0).astype(np.float64)
    Y_r = PCA.transform(Y_pred.reshape(-1, image_rows*image_cols))
    mask = PCA.inverse_transform(Y_r).reshape(image_rows, image_cols)
    mask[trans.resize(mask, (image_rows, image_cols)) > 0.4] = 1
    mask[mask != 1] = 0
    
    return mask

#%%
"""
Save all prediction images in 'predict_resuolts' folder in save_dir

pred, x_val , y_val and mm should be in the same order
pred: prediction result
mm: calcuated in ensemble method in main function, pixel to mm result in this list
"""
def save_pred_results(pred, x_val, y_val, mm, save_dir):
    
    for num in range(pred.shape[0]):
        dice = dice_coef_np(y_val[num,:,:,:],pred[num,:,:,:])
        s1 = sensitivity(y_val[num,:,:,:],pred[num,:,:,:])
        dist = mm[num]
        
        plt.figure(figsize=(15,10))

        plt.subplot(131)
        plt.title('Input')
        plt.imshow(x_val[num,0, :, :],cmap='gray')

        plt.subplot(132)
        plt.title('Ground Truth')
        plt.imshow(y_val[num,0, :, :],cmap='gray')
        
        plt.subplot(133)
        plt.title('DC:{} SEN:{} dist:{}'.format(round(dice,2), round(s1,2), round(dist,2)))
        plt.imshow(pred[num, 0, :, :],cmap='gray')

        plt.savefig(save_dir+'/{}.png'.format(num))
        plt.close()
        
#%% 
#======================MAIN FUNCTION=====================#       
"""
Preprocessing

gt_type:
a. 'kmuh': labeled by kmuh
b. 'new': labeled by MIRDC

create_origin (origin.py): crop and resize images to 496x592 (ex: 0000_cut.bmp)
create_GT: (GT.py): produce ground truth binary label (ex: 0000_mask.bmp)
wiener_filter (Filter.py): produce filtered images (ex: 0000_f.bmp)
create_data: read in and create npy file for training and validation data in 'npy' folder

NOTE: comment out these four function if data already prepared
"""
gt_type = 'new'

#create_origin(load_dir, '**/**.bmp')
#create_GT(load_dir, '**/**.bmp', gt=gt_type)
#wiener_filter(load_dir, '**/**.bmp', filter_size=7)
#create_data(load_dir, '**/**.bmp', label=False, gt=gt_type, resize=(image_rows,image_cols))

x_train = np.load(save_dir+'npy/'+gt_type+'/img_train.npy')
y_train = np.load(save_dir+'npy/'+gt_type+'/img_train_mask.npy')
x_val = np.load(save_dir+'npy/'+gt_type+'/img_val.npy')
y_val = np.load(save_dir+'npy/'+gt_type+'/img_val_mask.npy')

#%%
"""
Train model

train_model(x_train, y_train, x_val, y_val, model_layer, epoch, save_weight_dir)
load_weight: load in best weight in 'weight' folder
"""
np.random.seed(5)
np.random.shuffle(x_train)
np.random.seed(5)
np.random.shuffle(y_train)

model5 = unet_trim()
model4 = unet_4l()
model3 = unet_3l()

weight_dir = save_dir+'weight/'+gt_type+'/'
train_model(x_train, y_train, x_val, y_val, 5, 40, weight_dir)
train_model(x_train, y_train, x_val, y_val, 4, 200, weight_dir)
train_model(x_train, y_train, x_val, y_val, 3, 1000, weight_dir)

model5.load_weights(weight_dir+'5layer weight.h5')
model4.load_weights(weight_dir+'4layer weight.h5')
model3.load_weights(weight_dir+'3layer weight.h5')

#%%
"""
Ensemble, PCA method

x_result, y_result, pred_result and mm_result are in the same order (to evaluate output performance)
x_result: images in validation data
y_result: corresponding ground truth 
pred_result: final results
mm_result: central distance in mm unit 
PCA_label: determine use PCA method or not
save_pred_results: save all prediction images in 'pred_results' folder
"""
x_result=None
y_result=None
pred_result=None
mm_result=[]
PCA_label=True

for i in range(len(folders)):
    x_folder, y_folder = val_folder(load_dir, folders[i], '/**.bmp', gt=gt_type)
    
    if x_result==None:
        x_result = x_folder
        y_result = y_folder
    else:
        x_result = np.append(x_result, x_folder, axis=0)
        y_result = np.append(y_result, y_folder, axis=0)
        
    pred5 = model5.predict(x_folder, batch_size=12)
    pred4 = model4.predict(x_folder, batch_size=12)
    pred3 = model3.predict(x_folder, batch_size=12)
    
    ensemble = (pred5+pred4+pred3)/3
    
    ensemble[ensemble > 0.2] = 1
    ensemble[ensemble != 1] = 0 
    
    if PCA_label:
        PCA = pca(ensemble)
        newmask=[]
        for j in range(ensemble.shape[0]):
            img = ensemble[j].squeeze()
            mask = correct_mask(img)
            mask = mask.reshape((1,)+mask.shape)
            mask = mask.reshape((1,)+mask.shape)
            
            newmask.append(mask[0,:,:,:])
        newmask = np.array(newmask)
        
        dist = count_dist(y_folder, newmask)
        for k in range(newmask.shape[0]):
            pixel2mm = dist[k] * mm[i]
            mm_result.append(pixel2mm)
        
        if pred_result==None:
            pred_result = newmask
        else:    
            pred_result = np.append(pred_result, newmask, axis=0)
    
    else:
        dist = count_dist(y_folder, ensemble)
        for k in range(ensemble.shape[0]):
            pixel2mm = dist[k] * mm[i]
            mm_result.append(pixel2mm)
        
        if pred_result==None:
            pred_result = ensemble
        else:    
            pred_result = np.append(pred_result, ensemble, axis=0)

pred_dir = save_dir + 'predict_results'
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

save_pred_results(pred_result, x_result, y_result, mm_result, pred_dir)

#%%
"""
Evaluate output performance (function in performance.py)

draw_dice_hist: save dice coefficient histogram for prediction result in 'predict_result' folder
confusion: evaluate confusion matrix for prediction result
draw_confusion: calcuate accuracy and save image in 'predict_results' folder
draw_dist: calcuate average mm central distance in 'predict_results' folder
"""
pred_dc = draw_dice_hist(y_result, pred_result, save_dir)
pred_con = confusion(y_result, pred_result)
acc = draw_confusion(pred_con, save_dir)
pred_dist = draw_dist(mm_result, save_dir)
print('dice: ', pred_dc)
print('acc: ', acc)
print('dist: ', pred_dist)

#%%
"""
Figures in the paper

Figure 5-1: run wiener_test.py
Figure 5-5: set PCA_label to False to produce result without using PCA
Figure 5-7: function train_model(), save in 'weight' folder
Figure 5-8(a): function draw_dice_hist(), save in 'predict_results' folder
Figure 5-8(b): function draw_confusion(), save in 'predict_results' folder
Figure 5-9, 5-10: function save_pred_results(), save in 'predict_results' folder
Figure 5-11: comment out the ensemble method line can produce single u-net result
"""
