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
# time_in = time.perf_counter()

Worm = cv.VideoCapture(os.path.join(dirname, filename))

# Get frame count
num_frames = int(Worm.get(cv.CAP_PROP_FRAME_COUNT))
# print('No. of frames in video: ', num_frames)

# Get frame dimensions
w, h =  int(Worm.get(cv.CAP_PROP_FRAME_WIDTH)), int(Worm.get(cv.CAP_PROP_FRAME_HEIGHT))
# print('Frame width: ', w)
# print('Frame height', h) 

#%%

### Generate variance image

def var_img(vid):
    
    num_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

    # First generate the mean image
    Mean = 0
    count = 0
    while(vid.isOpened()):
        ret1, frame1 = vid.read()
        if ret1 == True:            
            Mean += np.float32(cv.cvtColor(frame1, cv.COLOR_BGR2GRAY))
            count += 1
            if count == num_frames:         # stopping condition
                break
    Mean = Mean/count

    # Now compute variance image
    Var = 0
    count = 0
    while(vid.isOpened()):
        ret1, frame1 = vid.read()
        if ret1 == True:            
            Var += (np.float32(cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)) - Mean)**2
            count += 1
            if count == num_frames:         # stopping condition
                break
    Var = Var/count
        
    # img_var_scaled = np.uint8(img_var)

    # ret, img_thresh = cv.threshold(img_var_scaled, 180, 255, cv.THRESH_BINARY)

    plt.figure()
    plt.subplot(121)
    plt.imshow(Mean, cmap='gray')
    plt.subplot(122)
    plt.imshow(Var, cmap='gray') 
    

time_in = time.perf_counter()

# Worm1 = cv.VideoCapture(os.path.join(dirname, filename))
var_img(Worm)

time_out = time.perf_counter()
print('Runtime in secs: ', (time_out-time_in))

#%%

### Select pillars the worm comes in contact with

# def pillar_worm_select(img_crop_store_f, k=0, thresh=10):

#     # k = 0                              # pick index for a non-worm pillar image
#     img_ref = img_crop_store_f[k]
#     # plt.imshow(img_ref.reshape((h_crop, w_crop)), cmap='gray');

#     # For all other images, get similarity in the form of Euclidean distance. If 
#     # this distance exceeds a certain threshold, we will store those pillar images 
#     # as potential candidates.

#     # sim = np.zeros((len(img_crop_store_f)-1,), dtype='uint8')
#     img_pillar_sel = np.zeros((len(img_crop_store_f),img_crop_store_f.shape[1]), 
#                           dtype='uint8')
#     img_pillar_nonworm = np.zeros((len(img_crop_store_f),img_crop_store_f.shape[1]), 
#                           dtype='uint8')

#     q = 0
    
#     ind_sel = []            # list for storing selcted indices
    
#     for p in range(1, len(img_crop_store_f)):
#         dist_euc = np.linalg.norm(img_crop_store_f[p,:]/255 - img_ref/255)
#         if dist_euc > thresh:
#             img_pillar_sel[q,:] = img_crop_store_f[p]
#             ind_sel.append(p)
#         else:
#             img_pillar_nonworm[q,:] = img_crop_store_f[p]
#         q += 1
        
#     # Deleting rows with zeros
#     img_pillar_sel_f = img_pillar_sel[~np.all(img_pillar_sel == 0, axis=1)]
#     img_pillar_nonworm_f = img_pillar_nonworm[~np.all(img_pillar_nonworm == 0, axis=1)]

#     # Display selected pillars that the worm comes in contact with
#     vec_len = img_crop_store_f.shape[1]
#     h_crop, w_crop = int(np.sqrt(vec_len)), int(np.sqrt(vec_len))
    
#     # Displaying/saving candidate pillar images 
#     for i in range(len(img_pillar_sel_f)):
#         img = img_pillar_sel_f[i,:].reshape((h_crop, w_crop))
#         # cv.imwrite('Pillar_sel_' + str(i) + '.jpg', img)
#         # plt.imshow(img_pillar_sel_f[i,:].reshape((h_crop, w_crop)), cmap='gray')
#         # plt.show()
    
#     # Display non-woprm pillars
#     # for j in range(len(img_pillar_nonworm_f)):
#     #     plt.imshow(img_pillar_nonworm_f[j,:].reshape((h_crop, w_crop)), cmap='gray')
#     #     plt.show()
    
#     return np.array(ind_sel), img_pillar_sel_f, img_pillar_nonworm_f

#%%

def img_worm_isolate(img, img_ref, k_dil=5, num_iter=1, t1=100, k_sz=35, 
                     a_crop=100, t2=80):
    
    # Worm isolate image - 1
    Frame_cur = np.abs(img.astype(np.float32) - img_ref.astype(np.float32))
    kernel_dil = np.ones((k_dil,k_dil), np.uint8)
    img_dil1 = cv.dilate(Frame_cur, kernel_dil, iterations=num_iter)
    thresh1, img_t1 = cv.threshold(img_dil1, t1, 255, cv.THRESH_BINARY)
    
    
    # Worm isolate image - 2
    img_n = 255 - img
    
    #Cropping out unwanted pillar near left edge
    # img_crop = img_n[:,a_crop:img_n.shape[1]]
    
    # Morphological opening to remove debris
    kernel = np.ones((k_sz,k_sz), np.uint8)
    img_op = cv.morphologyEx(img_n, cv.MORPH_OPEN, kernel)
    
    # Dilation to thicken worm - this will help in pillar selection
    kernel_dil = np.ones((k_dil,k_dil), np.uint8)
    img_dil2 = cv.dilate(img_op, kernel_dil, iterations=num_iter)
    
    # Simple Thresholding
    thresh2, img_t2 = cv.threshold(img_dil2, t2, 255, cv.THRESH_BINARY)
    
    ### Plotting ..
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(132)
    # plt.imshow(img_t1, cmap='gray')
    # plt.subplot(133)
    # plt.imshow(img_t2, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(img_t, cmap='gray')
    # plt.show()
    
    return img_t1, img_t2
    
                                          
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
            # x1, x2 = x_c - (h_crop//2), x_c + (h_crop//2) 
            # y1, y2 = y_c - (w_crop//2), y_c + (w_crop//2) 
            # im_crop = im[y1:y2, x1:x2]
            # im_crop_flatten = np.ravel(im_crop)
            # img_crop_store[i,:] = im_crop_flatten
            # plt.figure()
            # plt.imshow(img_crop_store[i,:].reshape((h_crop, w_crop)), cmap='gray')
            # plt.show()

    # Deleting rows with zeros
    # img_crop_store_f = img_crop_store[~np.all(img_crop_store == 0, axis=1)]
    
    return np.array(ind)

#%%

# Select pillars based on worm motion
# For each selected pillar, search neighborhood. If variance image signature 
# exists, store that pillar or else discard

def pillar_select_final(img, img_thresh1, img_thresh2, circles_crop, R=21, l_crop=100):
    
    x_c_list = circles_crop[:,0]
    y_c_list = circles_crop[:,1]
    rc_list = circles_crop[:,2]

    x_cf, y_cf, r_cf = [], [], []

    for i in range(len(x_c_list)):
        xc, yc, rc = int(x_c_list[i]), int(y_c_list[i]), int(rc_list[i])
        region1, region2 = img_thresh1[yc-R:yc+R, xc-R:xc+R], img_thresh2[yc-R:yc+R, xc-R:xc+R]
        if (np.max(region1) > 0) and (np.max(region2) > 0):
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
        
            # Cropping region near left frame edge to remove port
            # Frame_cur = Frame_cur[:,125:Frame_cur.shape[1]-100]

            # # Image processing to get binary image with isolated worm
            Frame_th1, Frame_th2 = img_worm_isolate(Frame_pres, Frame_prev)
        
            # Saving images
            # directory = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Images'
            # os.chdir(directory)
            # cv.imwrite(f'Segmented_img_1_{count}.jpg', Frame_th1)
            # cv.imwrite(f'Segmented_img_2_{count}.jpg', Frame_th2)
                
            X_cf, Y_cf, R_cf = pillar_select_final(Frame_cur_in, Frame_th1, 
                                      Frame_th2, circles_crop=circles_crop)
        
            X_p.append(X_cf), Y_p.append(Y_cf), R_p.append(R_cf)
        
            # Number of selected pillars
            n_pill += len(X_p)
        
            # Displaying selected pillars
            # fig,ax = plt.subplots(1)
            # ax.set_aspect('equal')

            # ax.imshow(Frame_pres, cmap='gray')
            # ax.axis('off')
        
            # Now, loop through coord arrays, and create a circle at each x,y pair
            # for xx, yy, rr in zip(X_cf, Y_cf, R_cf):
            #     circ = Circle((xx,yy), rr, color='red', fill=False, lw=0.5)
            #     ax.add_patch(circ)
            # plt.show()
        
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
            
            del(Frame_pres, Frame_prev, Frame_th1, Frame_th2, X_cf, Y_cf, R_cf)
        
        count += 1
        if count == num_frames:         # stopping condition
            break

#%%
### Remove repeated pillar detections across video frames
Frame_id_flat = [item for sublist in Frame_id.values() for item in sublist]
Result_valsX_flat = [item for sublist in Result_X.values() for item in sublist]
Result_valsY_flat = [item for sublist in Result_Y.values() for item in sublist]
Result_valsR_flat = [item for sublist in Result_R.values() for item in sublist]

Frame_id_final, X_final, Y_final, R_final = [], [], [], []
for i in range(len(Result_valsX_flat)):
    f_id, x, r = Frame_id_flat[i], Result_valsX_flat[i], Result_valsR_flat[i]
    for j in range(len(Result_valsY_flat)):
        y = Result_valsY_flat[j]
        if (x not in X_final) and (y not in Y_final):
            Frame_id_final.append(f_id)
            X_final.append(x)
            Y_final.append(y)
            R_final.append(r)

### Saving as Pandas dataframe
dict_res = {}
dict_res['Frame_id'] = Frame_id_final
dict_res['X_pillar'] = X_final
dict_res['Y_pillar'] = Y_final
dict_res['R_pillar'] = R_final

Results_df = pd.DataFrame(dict_res)

print('No. of final detected pillars: ', len(Results_df))

#%% Displaying final detections
        
### Displaying final pillars        
# for k in range(num_frames):
#     fig,ax = plt.subplots(1)
#     ax.set_aspect('equal')
    
#     ax.imshow(Frames[:,:,k], cmap='gray')
#     ax.axis('off')
    
#     for xx1, yy1, rr1 in zip(X_final, Y_final, R_final):
#         circ1 = Circle((xx1,yy1), rr1, color='red', fill=False, lw=0.5)
#         ax.add_patch(circ1)
#     # plt.show()
    
#     # Saving images
#     # directory = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Images/6002'
#     # os.chdir(directory)
#     # plt.savefig('Unique_pillars.png')
        
#%%

# Time out
time_out = time.perf_counter()
print('Runtime in secs: ', (time_out-time_in))

#%%
# Testing worm segmentation

# for i in range(5):
#     img_th = img_worm_isolate(Frames[:,:,i])
    
#     # plt.figure()
#     # plt.imshow(img_th, cmap='gray')
#     # plt.show()
    
#     del(img_th)

# img_th = img_worm_isolate(img)

#%%

# Detect pillars in individual images
# img1 = Frames[:,:,45]

# circles = detect_pillars(img1)

    
# # Plotting distribution of detected diameters
# plt.figure()
# plt.hist(2*circles[:,2], density=True)
# plt.xlabel('Diameter of detected circle in pixels')
# plt.ylabel('Probability')
# plt.xticks(np.linspace(0, 50, 6))
# # plt.yticks(np.linspace(0, 0.5, 6))
# plt.show()
    





    


    



        
