#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 00:15:04 2023

@author: anirudhgangadhar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 09:19:55 2023

@author: anirudhgangadhar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:52:13 2023

@author: anirudhgangadhar
"""


# PILLAR DETECTION CODE - FULLY AUTOMATED ANALYSIS!!!

# In this project, the dataset consists of brightfield videos containing C.Elegans
# worms moving inside a 3D microfluidic chamber consisting of a micropillar array.
# As the worms move, they contact certain pillars causing "pillar deflections".

# Pillars touched by the worm are selectively detected. Using the location of 
# detected centers, we can obtain the "deflection profile" of each selected pillar.

# A ".csv" file containing center pillar coordinates and frames the 
# pillar was contacted by the worm is also created. 

# The total processing time per video varies between <1 - 4 minutes. This is 
# dependent on FOV and number of frames in the video.

# Using "Numba" compiler to speed up function execution


### Importing libraries
# Standard libraries
import numpy as np
from itertools import chain
from numba import jit, vectorize
from pymlfunc import sub2ind
import scipy
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib import style
from itertools import count
from statistics import mode

# Image libraries
import cv2 as cv
from PIL import Image, ImageSequence
from skimage import morphology

# Miscellaneous libraries
import os
import time
import timeit
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')

# For creating dialog to allow user to select video file for analysis
import tkinter as tk
from tkinter import filedialog

#%%

### Pull up dialog box to let user select the appropriate video for analysis
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()

#%%
# test_code = '''

# Time in
time_in = time.perf_counter()

# Read video file - takes ~50 ms
Worm = cv.VideoCapture(filepath)

# Get frame count
num_frames = int(Worm.get(cv.CAP_PROP_FRAME_COUNT))
# Get frame dimensions
w, h =  int(Worm.get(cv.CAP_PROP_FRAME_WIDTH)), int(Worm.get(cv.CAP_PROP_FRAME_HEIGHT))

# Resampling factor - recommend setting this to "1" 0r "0.5" if min(FOV) 
# dimension < 1000 pixels
rsz = 0.2

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
    
    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv.CC_STAT_AREA]
        
        # remove connected components smaller than min_size
        if label_size < (np.around(min_size*(rsz**2))):
            img[labels == i] = 0
            
    return img

#%%

### Processing video to store frames in a 3D tensor
width, height = int(w * rsz), int(h * rsz)
dim = (width, height)
Frames = np.zeros((int(h*rsz), int(w*rsz), num_frames), dtype='uint8')    # initializing tensor to store video frames
count = 0

Thresh_stack = np.zeros((h, w, num_frames), dtype='uint8')  # initializing tensor to store stack of thresholded video frames 

while(Worm.isOpened()):
    ret, frame = Worm.read()    # reads a single video frame in RGB format
    if ret == True:
        frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # converting to grayscale image
        
        # Rescaling frame
        frame_g = cv.resize(frame_g, dim, fx=rsz, fy=rsz, interpolation = cv.INTER_NEAREST) 
        Frames[:,:,count] = frame_g
        
        count += 1
        if count == num_frames:         # stopping condition
            break

#%%

### MAIN ANALYSIS

### Circular pillar detection
Circles, radius_med = detect_pillars(Frames[:,:,0]) # use only first frame of video for detection
# Pill_cent = Pill_cent.astype('int32')

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

for j in range(num_frames): Mask_stack[:,:,j] = bwareaopen((Thresh_stack[:,:,j] 
                                                - Mat_seg_stack[:,:,j]))

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
for i in range(len(X_w)): Frames_w[i] = list(np.where(Pillar_w[X_w[i], Y_w[i], :] == 1))

#%%

### Next, we would like to obtain the deflection profiles of the selected pillars
sz_tile = int(25*rsz)
Mean_val = np.zeros((len(X_w), num_frames), dtype='float32')
Sd_val = np.zeros((len(X_w), num_frames), dtype='float32')

frame_ids = np.arange(0, num_frames)

for m in range(len(X_w)):
    frame_ids_sel = np.where(Pillar_w[X_w[m], Y_w[m], :] == 1)[0]
    for n in frame_ids:
        frame_w = Frames[X_w[m]-sz_tile:X_w[m]+sz_tile, 
                          Y_w[m]-sz_tile:Y_w[m]+sz_tile, n]
        Mean_val[m, n], Sd_val[m, n] = np.mean(frame_w)/255.0, np.std(frame_w)/255.0                
        
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

### Measure center coordinate location as a deflection metric
mid_pt = sz_tile + 1
Frame_nos, Xc_pillcropimg, Yc_pillcropimg = [], [], []

### Look at individual pillar cases
pillar = int(np.random.rand(1)*len(X_wr))
# pillar = 92
# pillar = 42
# pillar = 25
# pillar = 6
# pillar = 0 - no dislocation 

num_pillars_final = 0

for m in range(pillar, pillar+1):
    frame_ids_sel = np.where(Pillar_w[X_wr[m], Y_wr[m], :] == 1)[0]
    
    # Modifying frame_ids_sel array as we want to to consider +/- 10 frames
    frame_ids_sel_f = [list(np.arange(np.min(frame_ids_sel)-10, 
                      np.min(frame_ids_sel))), list(frame_ids_sel), 
                      list(np.arange(np.max(frame_ids_sel)+1, 
                      np.max(frame_ids_sel)+11))]
    
    frame_ids_sel_f = np.array(list(chain.from_iterable(frame_ids_sel_f)))
    
    # if (frame_ids_sel_f[0] > 10) & (frame_ids_sel[-1] < num_frames-11):
    
    for n in frame_ids_sel_f:
        frame_w = Frames[X_wr[m]-sz_tile:X_wr[m]+sz_tile+1, 
                          Y_wr[m]-sz_tile:Y_wr[m]+sz_tile+1, n]              
        
        ### Fit a circle to pillar in this image and get center location
        circles_cropimg = cv.HoughCircles(frame_w, cv.HOUGH_GRADIENT, 1, 10, 
                          param1=100, param2=5, minRadius=2, maxRadius=4)
        
        if circles_cropimg is not None:
        
            circles_cropimg = circles_cropimg.reshape((circles_cropimg.shape[1], 
                                                       circles_cropimg.shape[2]))
            
            # Storing (x_c, y_c) and 'r' of all detected circles
            x, y, r = circles_cropimg[:,0], circles_cropimg[:,1], circles_cropimg[:,2]
            
            # Calculate offset to correct centering of pillar image
            offset_x, offset_y = int(x - mid_pt), int(y - mid_pt)
            
            # Re-center pillar image - use offset
            frame_wc = Frames[X_wr[m]+offset_y-sz_tile:X_wr[m]+offset_y+sz_tile+1, 
                              Y_wr[m]+offset_x-sz_tile:Y_wr[m]+offset_x+sz_tile+1, n] 
            
            if len(frame_wc != 0):
            
                # Fit circle again
                circles_cropimg1 = cv.HoughCircles(frame_wc, cv.HOUGH_GRADIENT, 1, 
                                                   10, param1=100, param2=5, 
                                                   minRadius=2, maxRadius=4)
                
                if circles_cropimg1 is not None:
                
                    circles_cropimg1 = circles_cropimg1.reshape((circles_cropimg1.shape[1], 
                                                                 circles_cropimg1.shape[2]))

                    x1, y1, r1 = circles_cropimg1[:,0], circles_cropimg1[:,1], circles_cropimg1[:,2]
                    
                    # Store center coordinates for all frames of interest
                    Xc_pillcropimg.append(np.abs(np.mean(x1) - len(frame_wc)/2)) 
                    Yc_pillcropimg.append(np.abs(np.mean(y1) - len(frame_wc)/2)) 
                                   
                    Frame_nos.append(n)
                    
                    ### PLOT RESULTS
                    fig, axs = plt.subplots(2, 2, constrained_layout=True)
                    
                    axs[0,0].set_aspect('equal')
                    axs[0,0].imshow(frame_wc, cmap='gray')
                    axs[0,0].axis('off')
                    # axs[0,0].set_xticks([])
                    # axs[0,0].set_yticks([])
                    
                    axs[0,1].set_aspect('equal')
                    axs[0,1].imshow(frame_wc, cmap='gray')
                    axs[0,1].axis('off')
                    # axs[0,1].set_xticks([])
                    # axs[0,1].set_yticks([])
                
                    for xx, yy, rr in zip(x1, y1, r1):
                        circ = Circle((xx,yy), 0.1, color='red', fill=True, lw=1)
                        circ1 = Circle((xx,yy), rr, color='green', fill=False, lw=0.5)
                        axs[0,1].add_patch(circ)
                        axs[0,1].add_patch(circ1)
                    
                    ### Running plot - X_p/Y_p for every frame  
                    # X_p
                    axs[1,0].plot(Frame_nos, Xc_pillcropimg, '-o', color='blue', 
                              linewidth=0.2, ms=1)
                    axs[1,0].set_xlabel('Frame no.')
                    axs[1,0].set_ylabel('$D_{X}$')
                    axs[1,0].set_ylim([-0.05, np.max(Xc_pillcropimg)+0.05])
                    
                    frame_min, frame_max = np.min(frame_ids_sel), np.max(frame_ids_sel)
                    
                    # Draw a red vertical dotted line corresponding to first and last worm frame
                    frame_l, frame_u = frame_min*np.ones((100,)), frame_max*np.ones((100,))
                    y1 = np.linspace(np.min(Xc_pillcropimg)-0.05, np.max(Xc_pillcropimg)+0.05, 100)
                    axs[1,0].plot(frame_l, y1, color='red', linewidth=1)
                    y2 = np.linspace(np.min(Xc_pillcropimg)-0.05, np.max(Xc_pillcropimg)+0.05, 100)
                    axs[1,0].plot(frame_u, y2, color='red', linewidth=1)
                    
                    # Y_p
                    axs[1,1].plot(Frame_nos, Yc_pillcropimg, '-o', color='blue', 
                              linewidth=0.2, ms=1)
                    axs[1,1].set_xlabel('Frame no.')
                    axs[1,1].set_ylabel('$D_{Y}$')
                    axs[1,1].set_ylim([-0.05, np.max(Yc_pillcropimg)+0.05])
                    
                    # Draw a red vertical dotted line corresponding to first and last worm frame
                    y3 = np.linspace(np.min(Yc_pillcropimg)-0.05, np.max(Yc_pillcropimg)+0.05, 100)
                    axs[1,1].plot(frame_l, y3, color='red', linewidth=1)
                    
                    y4 = np.linspace(np.min(Yc_pillcropimg)-0.05, np.max(Yc_pillcropimg)+0.05, 100)
                    axs[1,1].plot(frame_u, y4, color='red', linewidth=1)
                    
                    plt.pause(0.0001)
    
#%%

# Check how many unique pillars are selected
# print('')
# print('No. of pillars contacted by worm: ', len(Pills_nz))

print('')
print('No. of final pillar selections: ', num_pillars_final)

# Stores results in dictionary        
Results = {}
Results['Pillar_center_X'], Results['Pillar_center_Y'], Results['Frames_worm contact'] = X_wr, Y_wr, Frames_wr

# Put results in a Pandas DataFrame
Results_df = pd.DataFrame(Results)

# Save as a ".csv" file
# os.chdir('/Users/anirudhgangadhar/Desktop/OPT volunteering/Worm project/Results')
# filename_noext = os.path.splitext(os.path.basename(filepath))[0]
# Results_df.to_csv(filename_noext + '.csv')

# '''

# Time out
time_out = time.perf_counter()
print('')
print('Runtime in secs: ', (time_out-time_in))

### Time out
# print('')
# print('Runtime in seconds: ', timeit.timeit(stmt=test_code, globals=globals(), number=1))
        
#%%


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

































