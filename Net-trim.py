# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:36:01 2020
Net-Trim 
@author: lin
"""
#information: https://dnntoolbox.github.io/Net-Trim/

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import time
import random as r
import skimage.transform as trans
from keras.models import Model

from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from keras import backend as K

import sys
import os
import copy
sys.path.append(os.path.abspath('/home/lin/Documents/Net-Trim-master/Main NetTrim Solvers'))
#import NetTrimSolver_tf as nt_tf
import ConvNetTrimSolver_tf as cnt_tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_image_dim_ordering("th")

#%%
smooth = 1. 
num_of_aug = 1
num_epoch = 40

image_rows = 496        
image_cols = 592        
folders = ['kvgh15_v06', 'kvgh05_01_v05', 'kmuh06_v15', 'kmuh05_v08', 'kmuh02_v18', 'kvgh07_01_v07', 'kvgh04_v08', 'kmuh12_v02']
num_of_val=8

load_dir = '/home/lin/Documents/test/part*/data/origin/' 
save_dir = '/home/lin/Documents/test/results/'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

    
def unet_model():
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

def cal_DC(y_val, pred):
    
    y_true_f = np.ndarray.flatten(y_val)
    y_pred_f = np.ndarray.flatten(pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
#%%
#load in training and validation data
x_train = np.load(save_dir+'npy/new/img_train.npy')
y_train = np.load(save_dir+'npy/new/img_train_mask.npy')
x_val = np.load(save_dir+'npy/new/img_val.npy')
y_val = np.load(save_dir+'npy/new/img_val_mask.npy')

#%%
#evaluate original 5-layer u-net performance
org_model = unet_model()
org_model.load_weights('/home/lin/Downloads/project_results/unet_GT/weights-project-best.h5')
org_pred = org_model.predict(x_val,batch_size=16)
org_dc = cal_DC(y_val, org_pred)
strides = [1, 1, 1, 1]

#%%
"""
org_weight: load in original weight
X: randomly pick feature maps at each layer
"""
org_weights = org_model.get_weights()

data_subset_size = 20
data_index = np.random.choice(np.arange(x_train.shape[0]), data_subset_size, replace=False)

inputs = [layer.output for layer in org_model.layers]
layer_inputs = K.function([org_model.input, K.learning_phase()], inputs)

X = layer_inputs([x_train[data_index, :, :, :], 0])

#%%
"""
See Table in Net-Trim.pdf
conv_layers: trim layer
conv_w_index, conv_b_index: corresponding weight, bias index
unroll_number: number of loops inside GPU
max_iterations: number of loops outside the GPU
perform cycle: max_iterations*unroll_number
"""
conv_layers = [0,1,3,4,6,7,9,10,12,13]
conv_w_index = [0,2,4,6,8,10,12,14,16,18]
conv_b_index = [1,3,5,7,9,11,13,15,17,19]

unroll_number = 2
max_iterations = 500
nt_conv = cnt_tf.NetTrimSolver(unroll_number=unroll_number, cg_iter=10, strides=strides, padding='SAME', precision=32)

#%%
"""
trim single layer, suggest trim one layer at each time
"""
conv_layers = [12]
conv_w_index = [16]
conv_b_index = [17]

unroll_number = 1
max_iterations = 100
nt_conv = cnt_tf.NetTrimSolver(unroll_number=unroll_number, cg_iter=10, strides=strides, padding='SAME', precision=32)

#%%
"""
Start trimming
epsilon_gain: larger to gain more sparsity, change in different layer
x,y: input and output feature maps

Parameter need to modify:
a. unroll_number
b. max_iteration
c. epsilon_gain  
"""
nt_weights = copy.deepcopy(org_weights)
epsilon_gain = 0.0001 
for k in range(len(conv_layers)):
    
    x = X[conv_layers[k]]
    y = X[conv_layers[k] + 1]
    W = org_weights[conv_w_index[k]]
    b = org_weights[conv_b_index[k]]
    
    x = np.transpose(x, axes=(0, 2, 3, 1))
    y = np.transpose(y, axes=(0, 2, 3, 1))
    
    V = np.zeros(y.shape)    
    norm_Y = np.linalg.norm(y)
    epsilon = epsilon_gain * norm_Y

    W_nt = nt_conv.run(x, y, V, b, W.shape, epsilon, rho=2, num_iterations=max_iterations)
    
    nt_weights[conv_w_index[k]] = W_nt
    nt_weights[conv_b_index[k]] = b

str_nnz = ', '.join('{}'.format(np.count_nonzero(np.abs(w) > 1e-4)) for w in nt_weights[::2])
print("number of non-zeros = {0}".format(str_nnz))

#%%
"""
Evaluate trim weight performance
weight dimension still remain the same
"""
new_model = unet_model()
new_model.set_weights(nt_weights)
new_pred = new_model.predict(x_val,batch_size=16)
new_pred[new_pred > 0.2] = 1 
new_pred[new_pred != 1] = 0 

new_dc = cal_DC(y_val, new_pred)

print('before:{}, after:{}'.format(round(org_dc,3), round(new_dc,3)))

#%% 
"""
Remove zero filter in the trim weight
th=0.005: if non-zero element < 0.5% => delete this filter, different threshold value when trimming different layer
example: remove conv5-1 filter size from 512 to 426
"""
new_weights = copy.deepcopy(nt_weights)
th = 0.005
layer=[16]  #correspond to conv_w_index
for _ in range(len(layer)):
    w = nt_weights[layer[_]]
    ft_size = w.shape[2]
    channel = w.shape[3]
    flat_w = w.reshape(9*ft_size,channel)
    flat_w = np.transpose(flat_w, axes=(1,0))
    
    z = 0
    tmp=[]
    zero_f=[]
    th = 3*3*ft_size*th
    for f in range(flat_w.shape[0]):
        s = np.count_nonzero(np.abs(flat_w[f]) > 1e-4)
        tmp.append(s)
        if (s<th):
            zero_f.append(f)
    
    delete_weights = np.delete(flat_w,zero_f[:],0)
    delete_weights = np.transpose(delete_weights, axes=(1,0))
    delete_weights = delete_weights.reshape(3,3,ft_size,channel-len(zero_f))
    print('remove shape: ', delete_weights.shape)
    print('remove filter: ', len(zero_f))

#Here we set conv5-1 with delete_weights and delete_b (bias)   
#Now the filter dimension in conv5-1: (3,3,256,512)->(3,3,256,426) 
new_weights[layer[0]] = delete_weights
delete_b = new_weights[layer[0]+1]
delete_b = np.delete(delete_b,zero_f[:],0)
new_weights[layer[0]+1] = delete_b


#conv5-2 need modify since the last layer (conv5-1) filter dimension has changed
#conv5-2: (3,3,512,512)->(3,3,426,512)
delete_w = org_weights[layer[0]+2]
print(delete_w.shape)
dim3 = delete_w.shape[2] - len(zero_f)
dim4 = delete_w.shape[3]
new_w = np.zeros((3,3,dim3,dim4))
for f in range(delete_w.shape[3]):
    tmp = delete_w[:,:,:,f]
    tmp2 = np.delete(tmp,zero_f[:],2)
    new_w[:,:,:,f] = tmp2

new_weights[layer[0]+2] = new_w

#%%
#Define the trim model
def unet_nt_model():
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)   # stride = 2

    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(116, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(233, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(426, (3, 3), activation='relu', border_mode='same')(pool4)
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
    
#%%
#test reduced weight outpur performance
tr_model = unet_nt_model()
tr_model.set_weights(new_weights)
tr_pred = tr_model.predict(x_val,batch_size=16)
tr_pred[tr_pred > 0.2] = 1 
tr_pred[tr_pred != 1] = 0 

tr_dc = cal_DC(y_val, tr_pred)

print('before:{}, after:{}'.format(round(org_dc,3), round(tr_dc,3)))

#%%
#Define the retrain model
def unet_new_model():
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)   # stride = 2

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

#%%
#Retrain model
np.random.seed(5)
np.random.shuffle(x_train)
np.random.seed(5)
np.random.shuffle(y_train)

rt_model = unet_new_model()
checkpoint = ModelCheckpoint('trim weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]
history = rt_model.fit(x_train, y_train, batch_size=12, validation_split=0.0, validation_data=(x_val, y_val) ,epochs = num_epoch,callbacks = callbacks_list ,verbose=1, shuffle=True)

#%%
#Evaluate retrain model output performance
rt_model.load_weights('/home/lin/trim weights.h5')
rt_pred = rt_model.predict(x_val,batch_size=16)
rt_pred[rt_pred > 0.2] = 1 
rt_pred[rt_pred != 1] = 0 

rt_dc = cal_DC(y_val, rt_pred)

print('before:{}, after:{}'.format(round(org_dc,3), round(rt_dc,3)))
