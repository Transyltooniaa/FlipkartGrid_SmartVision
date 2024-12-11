# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:23:56 2024

@author: jishu
"""
import cv2
import numpy as np
from Backend.Fruit_Freshness.Banana import mask as mp
from Backend.Fruit_Freshness.Banana import hue as hu
 
def run(image, score):
    
    # Case 1: Fresh Banana
    case_1 = """
    Fresh Banana
    Shelf-Life: 2-7 days (at room temperature)
    Characteristics: Firm texture, bright yellow color with a few small spots, sweet aroma, and no bruises or browning.
    Eatable or not: Definitely eatable.
    """
    
    # Case 2: Moderately Stale Banana
    case_2 = """
    Moderately Stale Banana
    Shelf-Life: 7-10 days (room temperature)
    Characteristics: Softer texture, increased browning or spotting on the peel, slightly mushy inside, and a more pronounced sweet flavor.
    Eatable or not: Eatable but may not be as enjoyable; best used in smoothies or baking.
    """
    
    # Case 3: Rotten Banana
    case_3 = """
    Rotten Banana
    Shelf-Life: Exceeds 10 days (room temperature)
    Characteristics: Very soft or mushy texture, dark brown or blackened peel, very strong, fermented smell, and signs of decay.
    Eatable or not: Not eatable.
    """

    
    original, image = mp.remove_background_grabcut(image)
    
    # Assume you already have a mask for the fruit (from previous steps)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound_1 = np.array([0, 0, 30])  # Example: Orange-ish fruit
    upper_bound_1 = np.array([180, 255, 100])
    binary_mask_1 = cv2.inRange(hsv_image, lower_bound_1, upper_bound_1)
    
    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    refined_mask_1 = cv2.morphologyEx(binary_mask_1, cv2.MORPH_CLOSE, kernel)
    refined_mask_1 = cv2.morphologyEx(refined_mask_1, cv2.MORPH_OPEN, kernel)
    
    # Detect low pixel values in the fruit area using the mask
    low_pixel_mask_1 = hu.detect_low_pixel_values(image, refined_mask_1)
    # print("Dark Pixels:", low_pixel_mask_1)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound_2 = np.array([10, 50, 100])  # Example: Orange-ish fruit
    upper_bound_2 = np.array([20, 200, 255])
    binary_mask_2 = cv2.inRange(hsv_image, lower_bound_2, upper_bound_2)
    
    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    refined_mask_2 = cv2.morphologyEx(binary_mask_2, cv2.MORPH_CLOSE, kernel)
    refined_mask_2 = cv2.morphologyEx(refined_mask_2, cv2.MORPH_OPEN, kernel)
    
    # Detect low pixel values in the fruit area using the mask
    low_pixel_mask_2 = hu.detect_low_pixel_values(image, refined_mask_2)
    # print("Brown Pixels:", low_pixel_mask_2)
    
    
    if low_pixel_mask_1 > 13:
        score = 0
    
    elif low_pixel_mask_2 > 20:
        score = 5
    else:
        score = 10
    
        
    answer = ""
    if score <=3 and score >= 0:
        answer = case_3
            
    elif score >= 4 and score <= 7:
        answer = case_2
    else:
        answer = case_1
        
    return answer
        
    
  