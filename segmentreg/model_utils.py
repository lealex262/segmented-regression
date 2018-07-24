'''
Written by Alex Le and Alex Chan

Helper functions to improve model preformance(Cleaning)
'''

import pandas as pd
import numpy as np
import random


# Adds new data points between actual data
def linear_augmentation(x, y, num_p = 2, init_weights = None, dim = '2d'):
    if '2d' == dim:
        # Setup for 2D
        size = len(x)
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        # Loops
        for i in range(size - 1):
            for j in range(num_p):
                # Weights
                if init_weights == None:
                    w = random.uniform(0, 1)
                else:
                    w = init_weights
                
                # Create Point
                val = (x[i] * w) + (x[i+1] * (1 - w))
                x = np.append(x, [val])
                val = (y[i] * w) + (y[i+1] * (1 - w))
                y = np.append(y, [val])

        return x, y


# Removes segments not larger that the threshold_size
def segment_fliter_by_size(segments, threshold_size = .05):
    new_segments = list()
    for i in range(len(segments)-1):
        if abs(segments[i][0] - segments[i+1][0]) > threshold_size:
            new_segments.append(segments[i])
    
    new_segments.append(segments[len(segments) - 1])
    return new_segments



# Merge segments that are closely related
def segment_merge(segments, threshold_diff = .05):
    indices = list()
    for i in range(len(segments)-1):
        if (abs(segments[i][1] - segments[i+1][1]) < threshold_diff):
            indices.append(i+1)
            
    for index in sorted(indices, reverse=True):
        del segments[index]
    
    return segments