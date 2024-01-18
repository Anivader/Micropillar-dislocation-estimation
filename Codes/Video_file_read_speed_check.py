#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:45:21 2023

@author: anirudhgangadhar
"""

## Importing libraries
# Standard libraries
import numpy as np
import scipy
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Image libraries
import cv2 as cv
from PIL import Image, ImageSequence
from skimage import morphology

# Miscellaneous libraries
import os
import time
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
# Time in
time_in = time.perf_counter()

# Read video file - takes ~50 ms
Worm = cv.VideoCapture(filepath)

# Get frame count
num_frames = int(Worm.get(cv.CAP_PROP_FRAME_COUNT))
# Get frame dimensions
w, h =  int(Worm.get(cv.CAP_PROP_FRAME_WIDTH)), int(Worm.get(cv.CAP_PROP_FRAME_HEIGHT))

# Resampling factor
rsz = 1

#%%

### MAIN ANALYSIS
width, height = int(w * rsz), int(h * rsz)
dim = (width, height)

Frames = np.zeros((int(h*rsz), int(w*rsz), num_frames), dtype='uint8')    # initializing tensor to store video frames
count = 0

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


# Time out
time_out = time.perf_counter()
print('')
print('Runtime in secs: ', (time_out-time_in))











