# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:01:27 2018

@author: lin

wiener filter 7*7
"""

import os
import skimage.io as io
import glob
from scipy import signal


def wiener_filter(src, mask, filter_size):
    files = glob.glob(src + mask, recursive=True)
    
    for file_name in files:
        save_name = file_name.replace('origin', 'filter')
        directory = os.path.split(save_name)
        dist = directory[0]
        if not os.path.exists(dist):
            os.makedirs(dist)
        if '_cut' in file_name:
            save_name = save_name.replace('_cut', '_f')
            img = io.imread(file_name, as_gray=True)
            img_f = signal.wiener(img, mysize=filter_size)
        
            #filter_name = file.replace('cut', 'filter')
            io.imsave(save_name, img_f)

if __name__=="__main__":
    wiener_filter(src, mask, filter_size)            