# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:06:01 2024

@author: jishu
"""

import cv2
import numpy as np

# Function to remove background using GrabCut
def remove_background_grabcut(image):
    
    if image is None:
        print("Error: Image not found!")
        return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a mask initialized to zero
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the background and foreground models (required by GrabCut)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object (this is an approximation)
    height, width = image.shape[:2]
    rect = (10, 10, width - 30, height - 30)  # Adjust the rectangle if necessary

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to separate foreground from background
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Refine the mask to remove noise (shadows, small regions, etc.)
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Apply the refined mask to the image
    foreground = image * refined_mask[:, :, np.newaxis]

    # Replace the background with white
    background = np.full_like(image, 255)  # Create a white background
    result = np.where(refined_mask[:, :, np.newaxis] == 1, foreground, background)

    return image, result

