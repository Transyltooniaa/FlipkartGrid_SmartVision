# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:15:04 2024

@author: jishu
"""

from Backend.Fruit_Freshness.Apples import shape as sp
from Backend.Fruit_Freshness.Apples import mask as mk
from Backend.Fruit_Freshness.Apples import hue as hu
from Database.mongo import DatabaseManager

import threading
import cv2
import numpy as np

db_manager = DatabaseManager()

    
def run(image, stale_flag=False, fresh_flag=False, score=10):
    
    # Case 1: Fresh Apple
    case_1 = """
    Fresh Apple
    Shelf-Life: 5-7 days (at room temperature)
    Characteristics: Firm texture, bright and vibrant color, no visible bruises or spots, a sweet and crisp taste.
    Eatable or not: Definitely eatable.
    """
    
    # Case 2: Moderately Stale Apple
    case_2 = """
    Moderately Stale Apple
    Shelf-Life: 8-14 days (room temperature) or 4 weeks (refrigerated)
    Characteristics : Slightly softer texture, slight discoloration or dull appearance, taste may be slightly sour but still acceptable.
    Eatable or not: Eatable but should be consumed soon.
    """
    
    # Case 3: Rotten Apple
    case_3 = """
    Rotten Apple
    Shelf-Life: Exceeds 14 days (room temperature) or 4 weeks (refrigerated)
    Characteristics: Mushy texture, dark or blackened spots, foul smell, and signs of mold or fermentation.
    Eatable or not: Not eatable.
    """
    
    # Case 4: Rotten Apple
    case_4 = """
    Rotten Apple
    Shelf-Life: Exceeds 14 days (room temperature) or 4 weeks (refrigerated)
    Characteristics: Mushy texture even though no blackened spots or fermentation, foul smell.
    Eatable or not: Not eatable.
    """
    # First Check for Colour COncentration
    orange_percentage, yellow_percentage = hu.find_orange_yellow_frequency(image)
    
    if not fresh_flag:
        if yellow_percentage > 50:
            score = 10
            fresh_flag = True
            
        
    if not fresh_flag:
        if orange_percentage > 10:
            score = 5
            
    # Then Check if there are wrinkles on the fruit
    block_size = (30, 30)  # Size of each block
    densities = sp.calculate_edge_density(image, block_size)
            
    if not fresh_flag:
        if densities > 1.2:
            score = score - 2

    # Assume you already have a mask for the fruit (from previous steps)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 100, 100])  # Example: Orange-ish fruit
    upper_bound = np.array([10, 150, 255])
    binary_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Detect low pixel values in the fruit area using the mask
    high_pixel_count = mk.detect_low_pixel_values(image, refined_mask)
    
    if not fresh_flag:
        if high_pixel_count > 8:
            stale_flag = True
    
    
    if stale_flag:
        score = 0
        
    answer = ""
    
    def update_db(fruit, shelf_life, characteristics, eatable):
        db_manager.add_freshness_record(fruit, shelf_life, characteristics, eatable)
    
    if score <=3 and score >= 0:
        if stale_flag:
            answer = case_3
            db_thread = threading.Thread(target=update_db, args=("Apple", "Exceeds 14 days (room temperature) or 4 weeks (refrigerated)", "Mushy texture, dark or blackened spots, foul smell, and signs of mold or fermentation.", "Not eatable"))
            db_thread.start()
            
        else:
            answer = case_4
            db_thread = threading.Thread(target=update_db, args=("Apple", "Exceeds 14 days (room temperature) or 4 weeks (refrigerated)", "Mushy texture even though no blackened spots or fermentation, foul smell.", "Not eatable"))
            db_thread.start()
            
    elif score >= 4 and score <= 7:
        answer = case_2
        db_thread = threading.Thread(target=update_db, args=("Apple", "8-14 days (room temperature) or 4 weeks (refrigerated)", "Slightly softer texture, slight discoloration or dull appearance, taste may be slightly sour but still acceptable.", "Eatable but should be consumed soon"))
        db_thread.start()
        
    else:
        answer = case_1
        db_thread = threading.Thread(target=update_db, args=("Apple", "5-7 days (at room temperature)", "Firm texture, bright and vibrant color, no visible bruises or spots, a sweet and crisp taste.", "Definitely eatable"))
        db_thread.start()
        
    return answer
        
    
        
        
        
        
            
    
    