#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:33:10 2023

@author: anirudhgangadhar
"""

import numpy as np
import cv2 as cv
import time
import timeit
from itertools import chain
from pymlfunc import sub2ind, ind2sub
from skimage import morphology

#%%
# def bwareaopen(img, min_size=10000, connectivity=8):
    
#     # Find all connected components (called here "labels")
#     num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
#         img, connectivity=connectivity)
    
#     # check size of all connected components (area in pixels)
#     # inds = np.where(stats[:,cv.CC_STAT_AREA] < min_size)[0]    
#     # img[labels == inds] = 0
    
    
#     # for i in range(num_labels):        
#     #     if (stats[i, cv.CC_STAT_AREA] < min_size): img[labels == i] = 0   # remove connected components smaller than min_size
    
#     return img_f

img = np.random.randint(2, size=(2160, 2560)).astype('uint8')
# img_f = bwareaopen(img)

img_f = morphology.remove_small_objects(img, min_size=10000, connectivity=8)

print('')
print(np.max(img_f - img))

#%%

# h, w, num_frames = 2160, 2560, 10

# arr1 = np.random.randint(2, size=(h, w, num_frames)).astype('uint8')
# arr2 = np.random.randint(2, size=(h, w, num_frames)).astype('uint8')

# arr3 = np.zeros((h, w, num_frames), dtype='uint8')

# # Time in
# time_in = time.perf_counter()

# # For loop
# for j in range(num_frames): arr3[:,:,j] = bwareaopen((arr1[:,:,j] - arr2[:,:,j]))

# # Time out
# time_out = time.perf_counter()
# print('')
# print('Runtime in secs: ', (time_out-time_in))


