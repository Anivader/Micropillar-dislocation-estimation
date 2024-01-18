#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:17:22 2022

@author: anirudhgangadhar
"""

# BASED ON WORM TRAVERSAL, THIS CODE WILL DETECT POTENTIAL PILLARS 
# OF INTEREST FOR COMPUTING DEFLECTIONS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
from PIL import Image, ImageSequence
from matplotlib.patches import Circle

import os
import time

# Set working path
dirname = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Videos/'

### Good detection videos
# filename = 'BZ33C_Chip1D_Worm27.avi'
# filename = 'BZ33C_Chip2A_Worm06.avi'

### False detection videos
# '9.avi'

# filename = '8.avi'
filename = '6002.avi'
# filename = '13.avi'
# filename = '14.avi'
# print(os.getcwd())

#%%
# Time in
time_in = time.perf_counter()

Worm = cv.VideoCapture(os.path.join(dirname, filename))

# Get frame count
num_frames = int(Worm.get(cv.CAP_PROP_FRAME_COUNT))
# print('No. of frames in video: ', num_frames)

# Get frame dimensions
w, h =  int(Worm.get(cv.CAP_PROP_FRAME_WIDTH)), int(Worm.get(cv.CAP_PROP_FRAME_HEIGHT))
# print('Frame width: ', w)
# print('Frame height', h) 
          
#%%

def img_worm_isolate(img, img_ref, k_dil=5, num_iter=1, t1=120):
    
    # Worm isolate image
    Frame_cur = np.abs(img.astype(np.float32) - img_ref.astype(np.float32))
    kernel_dil = np.ones((k_dil,k_dil), np.uint8)
    img_dil = cv.dilate(Frame_cur, kernel_dil, iterations=num_iter)
    thresh, img_t = cv.threshold(img_dil, t1, 255, cv.THRESH_BINARY)
    
    ### Plotting ..
    plt.figure()
    # plt.subplot(221)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(222)
    # plt.imshow(Frame_cur, cmap='gray')
    # plt.subplot(223)
    # plt.imshow(img_dil, cmap='gray')
    plt.subplot(111)
    plt.imshow(img_t, cmap='gray')
    plt.show()
    # plt.axis('off')
    
    return img_t
    
#%%
### Detect circular pillar structures

def detect_pillars(img):
    
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()
    
    # Detect circles using circular Hough transform
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=200, param2=20, 
                          minRadius=10, maxRadius=20)

    
    circles = circles.reshape((circles.shape[1], circles.shape[2]))

    ### Show detected circles in image
    # fig,ax = plt.subplots(1)
    # ax.set_aspect('equal')

    # Show the image
    # ax.imshow(img, cmap='gray')
    # ax.axis('off')
# 
    # Storing (x_c, y_c) and 'r' of all detected circles
    x, y, r = circles[:,0], circles[:,1], circles[:,2]

    # Now, loop through coord arrays, and create a circle at each x,y pair
    for xx, yy, rr in zip(x, y, r):
        circ = Circle((xx,yy), rr, color='red', fill=False, lw=0.4)
        # ax.add_patch(circ)
    # plt.show()
    
    # ax.scatter(x, y, c='red', marker='.', s=0.3)
    # plt.show()
    
    # Saving images
    # directory = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Images'
    # os.chdir(directory)
    # plt.savefig('Pillar_det.jpg')
    
    return circles

#%%

# Next, get indices of detected pillars, discard detection sclose to frame edges

def get_ind(im, circles):

    # img_crop_store = np.zeros((len(circles), h_crop * w_crop), dtype='uint8')
    ind = []        # array to store indices of selected detections       

    for i in range(len(circles)):
        x_c, y_c, r_c = circles[i]
        x_c, y_c, r_c = int(x_c), int(y_c), int(r_c)
    
        if (x_c > 100 and x_c < w-100) and (y_c > 100 and y_c < h-100):    
            ind.append(i)
    
    return np.array(ind)

#%%

# Select pillars based on worm motion
# For each selected pillar, search neighborhood. If variance image signature 
# exists, store that pillar or else discard

def pillar_select_final(img, img_thresh, circles_crop, R=21, l_crop=100):
    
    x_c_list = circles_crop[:,0]
    y_c_list = circles_crop[:,1]
    rc_list = circles_crop[:,2]

    x_cf, y_cf, r_cf = [], [], []

    for i in range(len(x_c_list)):
        xc, yc, rc = int(x_c_list[i]), int(y_c_list[i]), int(rc_list[i])
        region = img_thresh[yc-R:yc+R, xc-R:xc+R]
        if np.max(region) > 0:
            x_cf.append(xc)
            y_cf.append(yc)
            r_cf.append(rc)
            
    # Plotting 
    # fig,ax = plt.subplots(1)
    # ax.set_aspect('equal')

    # ax.imshow(img, cmap='gray')
    
    # Now, loop through coord arrays, and create a circle at each x,y pair
    for xx, yy, rr in zip(x_cf, y_cf, r_cf):
        circ = Circle((xx,yy), rr, color='red', fill=False, lw=0.5)
        # ax.add_patch(circ)
    # plt.show()

    return x_cf, y_cf, r_cf

#%%

### Main analysis

Frames = np.zeros((h, w, num_frames), dtype='uint8')
count = 0

count_pillar_sel = 0               # Initializing count of selected pillar candidates

# Initializing some lists of interest
Frame_id, X_p, Y_p, R_p = [], [], [], []

# Initializing pillar count
n_pill = 0

# matrix to store selcted pillar locations along with frame id
Frame_id, Result_X, Result_Y, Result_R = {}, {}, {}, {}
X_final, Y_final = [], []

while(Worm.isOpened()):
    ret, frame = Worm.read()
    if ret == True:
        frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        Frames[:,:,count] = frame_g
        Frame_pres = Frames[:,:,count]
        
        # Detecting circular pillars using CHT for first frame only
        if count == 0:
            Frame_cur_in = Frame_pres
            circles = detect_pillars(Frame_cur_in)
            ind = get_ind(Frame_cur_in, circles=circles)
            circles_crop = circles[ind, :]      # Get (x_c, y_c, r) of those circles for which cropped images were generated
        
        if count > 0:
            Frame_prev = Frames[:,:,count-1]    # store next frame for cleaning

            # # Image processing to get binary image with isolated worm
            Frame_th = img_worm_isolate(Frame_pres, Frame_prev)
        
            # Saving images
            # directory = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Images'
            # os.chdir(directory)
            # cv.imwrite(f'Segmented_img_1_{count}.jpg', Frame_th1)
            # cv.imwrite(f'Segmented_img_2_{count}.jpg', Frame_th2)
                
            X_cf, Y_cf, R_cf = pillar_select_final(Frame_cur_in, Frame_th, 
                                                   circles_crop=circles_crop)
        
            X_p.append(X_cf), Y_p.append(Y_cf), R_p.append(R_cf)
        
            # Number of selected pillars
            n_pill += len(X_p)
        
            # Displaying selected pillars
            fig,ax = plt.subplots(1)
            ax.set_aspect('equal')

            ax.imshow(Frame_pres, cmap='gray')
            ax.axis('off')
        
            # Now, loop through coord arrays, and create a circle at each x,y pair
            for xx, yy, rr in zip(X_cf, Y_cf, R_cf):
                circ = Circle((xx,yy), rr, color='red', fill=False, lw=0.5)
                ax.add_patch(circ)
            plt.show()
        
            # Saving images
            # directory = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Images/27001/Pillar selection'
            # os.chdir(directory)
            # plt.savefig(f'Pillar_select_{count}.png')
            
            # plt.close('all')

            # Get count of selected pillars
            count_pillar_sel += len(X_cf)
            
            # Store selected pillars in a dictionary
            Frame_id[count] = [count] * len(X_cf)
            Result_X[count] = X_cf
            Result_Y[count] = Y_cf
            Result_R[count] = R_cf
                        
            # Ensure that len(X) = len(Y)
            assert len(Result_X[(count)]) == len(Result_Y[(count)]), "Length of x-coord. entries not equal to y-coord"
            
            del(Frame_pres, Frame_prev, Frame_th, X_cf, Y_cf, R_cf)
        
        count += 1
        if count == 6:         # stopping condition
            break

#%%
### Remove repeated pillar detections across video frames
# Frame_id_flat = [item for sublist in Frame_id.values() for item in sublist]
# Result_valsX_flat = [item for sublist in Result_X.values() for item in sublist]
# Result_valsY_flat = [item for sublist in Result_Y.values() for item in sublist]
# Result_valsR_flat = [item for sublist in Result_R.values() for item in sublist]

# Frame_id_final, X_final, Y_final, R_final = [], [], [], []
# for i in range(len(Result_valsX_flat)):
#     f_id, x, r = Frame_id_flat[i], Result_valsX_flat[i], Result_valsR_flat[i]
#     for j in range(len(Result_valsY_flat)):
#         y = Result_valsY_flat[j]
#         if (x not in X_final) and (y not in Y_final):
#             Frame_id_final.append(f_id)
#             X_final.append(x)
#             Y_final.append(y)
#             R_final.append(r)

# ### Saving as Pandas dataframe
# dict_res = {}
# dict_res['Frame_id'] = Frame_id_final
# dict_res['X_pillar'] = X_final
# dict_res['Y_pillar'] = Y_final
# dict_res['R_pillar'] = R_final

# Results_df = pd.DataFrame(dict_res)

# print('No. of final detected pillars: ', len(Results_df))
        
#%%

# Time out
time_out = time.perf_counter()
print('Runtime in secs: ', (time_out-time_in))

    
    



        
