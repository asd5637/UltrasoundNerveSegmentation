# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:11:41 2020
output performance
@author: lin
"""
import numpy as np
import math
import matplotlib
from sklearn.decomposition import TruncatedSVD
smooth = 1.
image_rows = 496
image_cols = 592

#%%
def dice_coef_np(y_true, y_pred):
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def draw_dice_hist(y_result, pred_result, save_dir):
    dc_avg = dice_coef_np(y_result, pred_result)
    dc=[]
    for n in range(pred_result.shape[0]):
        dice = dice_coef_np(y_result[n,:,:,:],pred_result[n,:,:,:])
        dc.append(round(dice,2))
        
    plt.ylim(0,250)
    plt.grid(True)
    plt.hist(dc,bins=20)
    
    plt.title('dc histogram, average: {}'.format(round(dc_avg,3)))
    plt.savefig(save_dir+'predict_results/dc historgram.png')
    plt.show()
    
    return dc_avg
    
#%%
"""
function for central distance
"""
def neighbor(img,r,c):
    
    if img[r,c+1]==0 and img[r+1,c]==0 and img[r+1,c+1]==0 and img[r+1,c-1]==0:
        return 0
    else:
        return 1
        
def gt_center(img):
    c_gt = []
    center_r = 0
    center_c = 0
    pixel = 0
    for row in range(image_rows):
        for col in range(image_cols):
            if img[row,col] == 1:
                center_r+=row
                center_c+=col
                pixel+=1
    if pixel==0:
        c_gt.append([0,0])
    else:
        center_r/=pixel
        center_c/=pixel
        c_gt.append([int(center_r),int(center_c)])
    
    return c_gt  

def pred_center(img):
    c_pred = []
    num = 0
    center_r = 0
    center_c = 0
    pixel = 0
    for row in range(image_rows):
        for col in range(image_cols):
            if img[row,col] == 1:
                center_r+=row
                center_c+=col
                pixel+=1
                if neighbor(img,row,col)==0:    #redefine neighbor
                    num+=1
                    center_r/=pixel
                    center_c/=pixel
                    c_pred.append([int(center_r),int(center_c)])
                    center_r = 0
                    center_c = 0
                    pixel = 0
                    
    if num==0:
        c_pred.append([0,0])
    
    return c_pred    

def distance(x,y):
    X = x[0]-y[0]
    Y = x[1]-y[1]
    return round(math.sqrt(X**2+Y**2),3)    

def count_dist(y_val,pred):
    
    dist=[]
    
    for n in range(pred.shape[0]):
        img_gt = y_val[n,0,:,:]
        img_pred = pred[n,0,:,:]
        c_gt = gt_center(img_gt)
        c_pred = pred_center(img_pred)
        num = len(c_pred)
        
        minimum = 1000
        if c_gt[0]==[0,0] or c_pred[0]==[0,0]:
            dist.append(0)
        
        else:    
            for i in range(num):
                l = distance(c_gt[0],c_pred[i])
                if minimum>l:
                    minimum = l
            dist.append(minimum)
    return dist

def to_percent(y, position):
    s = str(10 * y)

    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def draw_dist(mm_result, save_dir):
    index = np.arange(len(mm_result))
    MM_result = [ round(elem,2) for elem in mm_result ]
    mean = np.mean(MM_result)
    
    plt.xlim((0,2))
    plt.xlabel('mm')
    plt.ylabel('percentage')
    plt.hist(MM_result,bins=20, range=(0,2),alpha=0.7, rwidth=0.85,normed=True)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title('mean distance: {} mm'.format(round(mean,3)))
    plt.savefig(save_dir+'predict_results/dist.png')
    plt.show()
    
    return mean
    
#%%
"""
function for confusion mat
"""
def img_confusion(y_img, p_img):
    y_one = 1 in y_img
    p_one = 1 in p_img
    overlap = np.sum(np.logical_and(y_img, p_img))   
    
    if y_one==True and p_one==True and overlap>0:
        return 'TP'
    elif y_one==True and p_one==True and overlap==0:
        return 'FN'
    elif y_one==False and p_one==False:
        return 'TN'
    elif y_one==False and p_one==True:
        return 'FP'
    elif y_one==True and p_one==False:
        return 'FN'

def confusion(y_val, pred):
    TP=0
    TN=0
    FP=0
    FN=0
    TP_case=[]
    TN_case=[]
    FP_case=[]
    FN_case=[]
    
    for n in range(pred.shape[0]):
        c = img_confusion(y_val[n,:,:,:],pred[n,:,:,:])
        
        if c=='TP':
            TP+=1
            TP_case.append(n)
        elif c=='TN':
            TN+=1
            TN_case.append(n)
        elif c=='FP':
            FP+=1
            FP_case.append(n)
        elif c=='FN':
            FN+=1
            FN_case.append(n)
            
    return TP,TN,FP,FN    

def draw_confusion(pred_con, save_dir):
    index = np.arange(4)
    plt.bar(index,pred_con,width=0.45,facecolor = 'lightskyblue')
    for x,y in zip(index,pred_con):
        plt.text(x+0.2, y+0.05, '%d' % y, ha='center', va= 'bottom')
    plt.xticks((0.25,1.25,2.25,3.25), ( 'TP', 'TN','FP', 'FN'))
    acc = (pred_con[0]+pred_con[1])/np.sum(pred_con)
    plt.title('acc: {}'.format(round(acc,3)))
    plt.savefig(save_dir+'predict_results/confusion.png')
    plt.show()
    
    return acc
