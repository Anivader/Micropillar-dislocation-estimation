#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:26:37 2023

@author: anirudhgangadhar
"""

import numpy as np
import matplotlib.pyplot as plt

X_ref, Y_ref, U, V = 0, 0, 1, 1

plt.quiver(X_ref, Y_ref, U, V, angles='uv', color='m', scale=5)