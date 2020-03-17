import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import math
import glob

def create_GT(src, mask, gt='new'):
    file_name = glob.glob(src + mask, recursive=True)
    
    for file in file_name:
        if '_cut' in file:
            continue
        if gt=='new':
            src_name = file.replace('origin', 'XX')
        else:
            src_name = file.replace('origin', 'KMUH')
        img = io.imread(src_name)
        if 'kmuh' in file:
            rows = 500
            cols = 600
            cut = img[50:550,100:700,:]
        else:
            rows = 750
            cols = 800
            cut = img[75:825,325:1125,:]
        GT = np.zeros((rows,cols))
        for r in range(rows):
            for c in range(cols):
                if cut[r,c,0] == 255 and cut[r,c,1] != 255 and cut[r,c,2] != 255:
                    GT[r,c] = 1
                else:
                    GT[r,c] = 0

        for r in range(rows):
            left = 0
            right = 0
            for c in range(cols):
                if GT[r,c] == 1 and left==0:
                    left = c
                elif GT[r,c]==1 and left!=0:
                    right = c
            if left!=0:
                GT[r,left:right+1] = 1
        
        GT_name = file.split('.')[0] + '_mask.bmp'
        s1 = GT_name.split('part')[0]
        s2 = GT_name.split('origin')[1]
        if gt=='new':
            GT_name = s1+'new_gt'+s2
        else:
            GT_name = s1+'kmuh_gt'+s2
        directory = os.path.split(GT_name)
        dist = directory[0]
        if not os.path.exists(dist):
            os.makedirs(dist)
        #GT_name = GT_name.replace('KMUH', 'origin')
    
        io.imsave(GT_name, GT)

if __name__=="__main__":
    create_GT(src, mask, gt='new')