#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:33:25 2023

@author: anirudhgangadhar
"""

# PILLAR DETECTION CODE - FULLY AUTOMATED ANALYSIS!!!

# In this project, the goal is to build an automated program that can select pillars
# contacted by the worm along with the specific frames this happened. Then, we 
# want to generate dislocation profiles for these selected pillars at the frames 
# of interest (foi).

# Dataset consists of brightfield videos containing C.Elegans
# worms moving inside a 3D microfluidic chamber consisting of a micropillar array.
# As the worms move, they contact certain pillars causing "pillar deflections".

# A ".csv" file containing center pillar coordinates and frames the 
# pillar was contacted by the worm is also saved. 

# The total processing time per video varies between <1 - 4 minutes. This is 
# dependent on FOV and number of frames in the video.

#%%

# RUN THIS CELL INDEPENDENTLY TO MAKE SURE YOU HAVE INSTALLED ALL THE 
# REQUIRED LIBRARIES FIRST

### Importing libraries
# Standard libraries
import numpy as np
from itertools import chain
from numba import jit, vectorize
from pymlfunc import sub2ind, ind2sub
import scipy
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib import style
from itertools import count

# Image libraries
import cv2 as cv
from PIL import Image, ImageSequence
from skimage import morphology

# Miscellaneous libraries
import os
import time
import timeit

# For creating dialog to allow user to select video file for analysis
import tkinter as tk
from tkinter import filedialog

#%%

### Pull up dialog box to let user select the appropriate video for analysis
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()

#%%
# # Time in
# time_in = time.perf_counter()

# Read video file - takes ~50 ms
Worm = cv.VideoCapture(filepath)

# Get frame count
num_frames = int(Worm.get(cv.CAP_PROP_FRAME_COUNT))
# Get frame dimensions
w, h =  int(Worm.get(cv.CAP_PROP_FRAME_WIDTH)), int(Worm.get(cv.CAP_PROP_FRAME_HEIGHT))

# Downsampling factor and new frame dimensions
rsz = 0.2
width, height = int(w * rsz), int(h * rsz)
dim = (width, height)

#%%
### Function to convert video object to stack of frames
@jit
def vid2framestack(Worm):
    Frames = np.zeros((int(h*rsz), int(w*rsz), num_frames), dtype='uint8')    # initializing tensor to store video frames
    Frames_fullsz = np.zeros((h, w, num_frames), dtype='uint8')
    count = 0
    while(Worm.isOpened()):
        ret, frame = Worm.read()    # reads a single video frame in RGB format
        if ret == True:
            frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # converting to grayscale image
            frame_g_rsz = cv.resize(frame_g, dim, fx=rsz, fy=rsz, 
                                    interpolation = cv.INTER_NEAREST)   # rescaling frame
            Frames[:,:,count], Frames_fullsz[:,:,count] = frame_g_rsz, frame_g
            count += 1
            if count == num_frames:         # stopping condition
                break
    return Frames, Frames_fullsz

#%%
### Detect circular pillar structures
@jit
def detect_pillars(img):
        
    # Detect circles using circular Hough transform
    # Apply twice - first for coarse detections and second for more stringent ones
    # More lenient radius range
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, int(50*rsz), param1=200, param2=5, 
                          minRadius=int(10*rsz), maxRadius=int(20*rsz))
    circles = np.around(circles.reshape((circles.shape[1], circles.shape[2])))
    
    # Get median radius of detected circles
    radius_med = int(np.around(np.median(circles[:,2])))    # get median radius value
            
    return circles, radius_med

#%%
# Remove debris/junk present in video frames
@jit
def bwareaopen(img, min_size=10000, connectivity=8):
    
    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        img, connectivity=connectivity)
    
    # # remove connected components smaller than min_size
    for i in range(num_labels):
        if stats[i, cv.CC_STAT_AREA] < np.around(min_size*(rsz**2)): img[labels == i] = 0
            
    return img

#%%
### WORM PILLAR DETECTION

### Converting video object to 3d tensor of frames
Frames, Frames_fullsz = vid2framestack(Worm)

### Circular pillar detection
Circles, radius_med = detect_pillars(Frames[:,:,0]) # use only first frame of video for detection

# Storing "X" and "Y" coordinates of detected pillar centers in matrix
Pill_cent = Circles[:,:-1].astype(np.int32)
X_p, Y_p = list(Pill_cent[:,0]), list(Pill_cent[:,1])

# Create a binary image with "0's" in all locations corresponding to detected pillars
Frame_bin = np.zeros((height*width,), dtype='uint8')

ind1 = sub2ind([height, width], Pill_cent[:,1], Pill_cent[:,0])
Frame_bin[ind1] = 1
Frame_bin = Frame_bin.reshape((height, width))

# Create frame stack by repeating binary frame "num_frames" times
Frame_bin_stack = np.repeat(Frame_bin, repeats=num_frames, axis=1)
Frame_bin_stack = Frame_bin_stack.reshape((height, width, num_frames))

### Creating skin-tight mask around worm
# First, threshold the stack of video frames - OTSU
Thresh_stack = np.zeros((height, width, num_frames), dtype='uint8')
for i in range(num_frames):
    ret, Thresh_stack[:,:,i] = cv.threshold(Frames[:,:,i], 0, 1, cv.THRESH_BINARY+cv.THRESH_OTSU)   # max value of "1" gives better result than "255" but this is because of rounding-off error
    Thresh_stack[:,:,i] = 1 - Thresh_stack[:,:,i]
    del(ret)
Thresh_stack_b = Thresh_stack.astype(bool)

# Summing over all frames
Mat_seg = np.sum(Thresh_stack_b, axis=2, keepdims=True) - num_frames + 1
Mat_seg[Mat_seg <= 0] = 0
Mat_seg_b = Mat_seg.astype(bool)

Mat_seg_stack = np.repeat(Mat_seg_b, repeats=num_frames, axis=1)
Mat_seg_stack = Mat_seg_stack.reshape((height, width, num_frames))

# Removing small objects/debris - "bwareaopen" better than "morphological opening"
Mask_stack = np.zeros((height, width, num_frames), dtype='uint8')
for j in range(num_frames): Mask_stack[:,:,j] = bwareaopen((Thresh_stack[:,:,j] - Mat_seg_stack[:,:,j]))

### Obtain worm pillars
kernel_dil = morphology.disk(np.around(3*radius_med))
Mask_dil = cv.dilate(Mask_stack, kernel_dil, iterations=1)

# Get worm pillars for all frames in video
Pillar_w = np.multiply(Frame_bin_stack, Mask_dil)

#%%
### Extract data and save results

# Use summing method 
# Sum across all video frames and consider pillar only if worm touched it
# Consider the fact that worm can touch the pillar for a few frames, then not 
# touch it and then touch it again. This can happen multiple times.

Counts = np.sum(Pillar_w, axis=2)

# Let us only consider those pillars which contacted the worm for > 5 frames.
# We can also impose an upper limit here (<20). This will take care of false pillar 
# detections
X_nw, Y_nw = np.where(Counts < 5)
for x_nw, y_nw in zip(X_nw, Y_nw): Counts[x_nw, y_nw] = 0

# Now find all locations corresponding to selected pillar locations
X_w, Y_w = np.where(Counts.astype(bool) == 1)
        
# Put "1" only at those locations where selected pillar exists in "Pillar_w" array
for x_nw1, y_nw1 in zip(X_nw, Y_nw): Pillar_w[x_nw1, y_nw1, :] = 0

# We want to get all frames where the worm comes in contact with a selected
# pillar and store it
Frames_w = np.zeros((len(X_w), ), dtype=object)
for i in range(len(X_w)): Frames_w[i] = list(np.where(Pillar_w[X_w[i], 
                                              Y_w[i], :] == 1))

#%%

### Get mean value of all pixels in every pillar image
sz_tile = int(25*rsz)
Mean_val = np.zeros((len(X_w), num_frames), dtype='float32')

frame_ids = np.arange(0, num_frames)

for m in range(len(X_w)):
    frame_ids_sel = np.where(Pillar_w[X_w[m], Y_w[m], :] == 1)[0]
    for n in frame_ids:
        frame_w = Frames[X_w[m]-sz_tile:X_w[m]+sz_tile, 
                          Y_w[m]-sz_tile:Y_w[m]+sz_tile, n]
        Mean_val[m, n] = np.mean(frame_w)/255.0              
        
    del(frame_ids_sel)
    
#%%

# Remove falsely detected pillars - SD criterion
# Basically, is the pillar is contacted by the worm, there will be a significant 
# drop in the mean intensity value of that pillar image.
Mean_val_filt = np.zeros((len(X_w), num_frames), dtype='float64')
Pills_nz = np.zeros((len(X_w),), dtype='int')
for j in range(len(Mean_val)):
    if np.std(Mean_val[j,:]) > 2e-2: 
        Mean_val_filt[j,:] = Mean_val[j,:]
        Pills_nz[j] = j
Mean_val_filt = Mean_val_filt[Mean_val_filt != 0].reshape((-1,num_frames))
Pills_nz = Pills_nz[Pills_nz != 0]

# Now store centers and frames of "real" pillars only
X_wr, Y_wr, Frames_wr = X_w[Pills_nz], Y_w[Pills_nz], Frames_w[Pills_nz]

#%%

### Get "(X_wr, Y_wr)" in original unscaled image
X_wr_unsc, Y_wr_unsc = (X_wr*int(h/height)), Y_wr*int(w/width)

### Get center coordinate location as a deflection metric
mid_pt = int(sz_tile/rsz) + 1
Frame_nos, dist, direction = [], [], []
X_pill, Y_pill, Radius = [], [], []

### Look at individual pillar cases
# pillar = int(np.random.rand(1)*len(X_wr_unsc))
# pillar = 23

num_pillars_final, count = 0, 0
num_frames_out = 10     # of frames you want to look at outside worm frame bounds

for m in range(len(X_wr_unsc)):
    X_pillsingle, Y_pillsingle = [], []
    frame_ids_sel = np.where(Pillar_w[X_wr[m], Y_wr[m], :] == 1)[0]
    
    # Modifying frame_ids_sel array as we want to to consider +/- 10 frames
    frame_ids_sel_f = [list(np.arange(np.min(frame_ids_sel)-num_frames_out, 
                      np.min(frame_ids_sel))), list(frame_ids_sel), 
                      list(np.arange(np.max(frame_ids_sel)+1, 
                      np.max(frame_ids_sel)+num_frames_out+1))]
    
    frame_ids_sel_f = np.array(list(chain.from_iterable(frame_ids_sel_f)))
    
    # Comment out this "if" statement if you want to look at individual cases
    if (frame_ids_sel[0] > num_frames_out) & (frame_ids_sel[-1] < num_frames-(num_frames_out+1)):
    
        frame_w = Frames_fullsz[X_wr_unsc[m]-int(sz_tile/rsz):X_wr_unsc[m]+int(sz_tile/rsz)+1, 
                          Y_wr_unsc[m]-int(sz_tile/rsz):Y_wr_unsc[m]+int(sz_tile/rsz)+1, frame_ids_sel_f[0]]
        
        ### Fit a circle to pillar in this image and get center location
        circles_cropimg = cv.HoughCircles(frame_w, cv.HOUGH_GRADIENT, 1, 
                                          10, param1=100, param2=30, 
                                          minRadius=int(2/rsz), maxRadius=int(4/rsz))
        
        if circles_cropimg is not None:
        
            circles_cropimg = circles_cropimg.reshape((circles_cropimg.shape[1], 
                                                        circles_cropimg.shape[2]))
            
            # Storing (x_c, y_c) and 'r' of all detected circles
            x, y, r = circles_cropimg[:,0], circles_cropimg[:,1], circles_cropimg[:,2]
            
            # Calculate offset to correct centering of pillar image
            offset_x, offset_y = int(x - mid_pt), int(y - mid_pt)
            
            X_new, Y_new = X_wr_unsc[m]+offset_y, Y_wr_unsc[m]+offset_x
            
        for n in frame_ids_sel_f:
            frame_wc = Frames_fullsz[X_new-int(sz_tile/rsz):X_new+int(sz_tile/rsz)+1, 
                              Y_new-int(sz_tile/rsz):Y_new+int(sz_tile/rsz)+1, n] 
            
            if len(frame_wc != 0):
            
                # Fit circle again
                circles_cropimg1 = cv.HoughCircles(frame_wc, cv.HOUGH_GRADIENT, 1, 
                                                    10, param1=100, param2=30, 
                                                    minRadius=int(2/rsz), maxRadius=int(4/rsz))
                
                if circles_cropimg1 is not None:
                    
                    Frame_nos.append(n)
                
                    circles_cropimg1 = circles_cropimg1.reshape((circles_cropimg1.shape[1], 
                                                                  circles_cropimg1.shape[2]))

                    x1, y1, r1 = circles_cropimg1[:,0], circles_cropimg1[:,1], circles_cropimg1[:,2]
                    
                    X_pill.append(x1)
                    Y_pill.append(y1)
                    Radius.append(r1)
                    
                    X_pillsingle.append(x1)
                    Y_pillsingle.append(y1)
                    
                    # Impose a radius filter here
                    # If "r" is below/above a threshold, we will consider previous "x", "y" & "r"
                    if (((Radius[-1] < 14.5) | (Radius[-1] > 15.5)) & (np.min([len(X_pill), len(Y_pill), len(Radius)]) > 1)).all():
                        # Radius[-1], r1, x1, y1 = Radius[-2], Radius[-2], X_pill[-2], Y_pill[-2]
                        Radius[-1], r1 = Radius[-2], Radius[-2]
                    
                    # Store center coordinates for all frames of interest
                    X_cur, Y_cur = np.mean(x1), np.mean(y1)
                    pt = np.array((X_cur, Y_cur))
                    
                    # Compute Euclidean distance between (X_p, Y_p) for a
                    # particular frame and (X_ref, Y_ref)
                    X_ref, Y_ref = X_pillsingle[0], Y_pillsingle[0]
                    pt_ref = np.array((X_ref, Y_ref))
                    
                    d = np.linalg.norm(pt - pt_ref)
                    dist.append(d)
                    
                    # Get deflection direction
                    angle = np.arctan((Y_cur - Y_ref)/(X_cur - X_ref))
                    
                    # Account for "0/0" cases
                    if np.isnan(angle): 
                        angle = 0.0
                        
                    direction.append(angle)
                    
                    ### PLOT RESULTS FOR RANDOMLY SELECTED PILLAR
                    # fig, axs = plt.subplots(2, 2, constrained_layout=True)
                    
                    # axs[0,0].set_aspect('equal')
                    # axs[0,0].imshow(frame_wc, cmap='gray')
                    # axs[0,0].axis('off')
                    # # axs[0,0].grid()
                    # axs[0,0].set_title('51x51 centered pillar image', size=12)
                    # # axs[0,0].set_xticks([0, 10, 30, 50])
                    # # axs[0,0].set_yticks([0, 10, 30, 50])
                    
                    # axs[0,1].set_aspect('equal')
                    # axs[0,1].imshow(frame_wc, cmap='gray')
                    # axs[0,1].axis('off')
                    # # axs[0,1].grid()
                    # axs[0,1].set_title('Fit circle with center')
                    # # axs[0,1].set_xticks([0, 10, 30, 50])
                    # # axs[0,1].set_yticks([0, 10, 30, 50])
                
                    # for xx, yy, rr in zip(x1, y1, r1):
                    #     circ = Circle((xx,yy), 0.1, color='red', fill=True, lw=1)
                    #     circ1 = Circle((xx,yy), rr, color='green', fill=False, lw=0.5)
                    #     axs[0,1].add_patch(circ)
                    #     axs[0,1].add_patch(circ1)
                    
                    # ### Dislocation magnitude  
                    # axs[1,0].plot(Frame_nos, dist, '-o', color='blue', 
                    #           linewidth=0.2, ms=1)
                    # # axs[1,0].scatter(Frame_nos, dist, s=3, c='blue', marker='o')
                    # axs[1,0].set_xlabel('Frame no.')
                    # axs[1,0].set_ylabel('Distance (pixels)')
                    # axs[1,0].set_ylim([np.min(dist)-0.05, np.max(dist)+0.05])
                    # axs[1,0].set_title('Dislocation magnitude')
                    
                    # frame_min, frame_max = np.min(frame_ids_sel), np.max(frame_ids_sel)
                    
                    # # Draw a red vertical dotted line corresponding to first and last worm frame
                    # frame_l, frame_u = frame_min*np.ones((100,)), frame_max*np.ones((100,))
                    # y1 = np.linspace(np.min(dist)-0.05, np.max(dist)+0.05, 100)
                    # axs[1,0].plot(frame_l, y1, color='red', linewidth=1)
                    # y2 = np.linspace(np.min(dist)-0.05, np.max(dist)+0.05, 100)
                    # axs[1,0].plot(frame_u, y2, color='red', linewidth=1)
                    
                    # # Dislocation direction
                    # axs[1,1].plot(Frame_nos, direction, '-o', color='blue', 
                    #           linewidth=0.2, ms=1)
                    # # axs[1,1].scatter(Frame_nos, direction, s=3, c='blue', marker='o')
                    # axs[1,1].set_xlabel('Frame no.')
                    # axs[1,1].set_ylabel('Orientation (deg.)')
                    # axs[1,1].set_ylim([np.min(direction)-0.05, np.max(direction)+0.05])
                    # axs[1,1].set_title('Dislocation direction')
                    
                    # # Draw a red vertical dotted line corresponding to first and last worm frame
                    # y3 = np.linspace(np.min(direction)-0.05, np.max(direction)+0.05, 100)
                    # axs[1,1].plot(frame_l, y3, color='red', linewidth=1)
                    
                    # y4 = np.linspace(np.min(direction)-0.05, np.max(direction)+0.05, 100)
                    # axs[1,1].plot(frame_u, y4, color='red', linewidth=1)
                    
                    # # ### Saving figures
                    # # directory = '/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Results_pillar dislocation/BZ33C_Chip1D_Worm27/Pillar 2_noisy'
                    # # os.chdir(directory)
                    # # plt.savefig(f'Result_{count}.tif')
                    # # plt.close('all') 
                    
                    # count += 1

                    # plt.pause(0.0001) 
                    
    # del(X_new, Y_new)

    num_pillars_final += 1    
                
del(frame_ids_sel, frame_ids_sel_f)
         
#%%

# Check how many unique pillars are selected
print('')
print('No. of pillars selected: ', num_pillars_final)

# Report maximum deflection magnitude observed
print('')
print('Max. deflection magnitude: ', np.max(dist))

# Report maximum deflection direction observed
# print('')
# print('Max. deflection direction: ', np.max(direction)*(180/np.pi))

# Stores results in dictionary - "(X_p, Y_p)" are reported for uncompressed frame        
Results = {}
Results['Pillar_center_X'], Results['Pillar_center_Y'], Results['Frames_worm contact'] = X_wr_unsc, Y_wr_unsc, Frames_wr

# Put results in a Pandas DataFrame
Results_df = pd.DataFrame(Results)

# Save as a ".csv" file
# os.chdir('/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Results_excelfile')
# filename_noext = os.path.splitext(os.path.basename(filepath))[0]
# Results_df.to_csv(filename_noext + '.csv')

# '''

# # Time out
# time_out = time.perf_counter()
# print('')
# print('Runtime in secs: ', (time_out-time_in))

#%%


