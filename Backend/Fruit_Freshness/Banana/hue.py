# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:30:18 2024

@author: jishu
"""

import cv2
import numpy as np
from Backend.Fruit_Freshness.Banana import hue as hu
from Backend.Fruit_Freshness.Banana import mask as mp

def detect_low_pixel_values(image, mask, threshold=50):
    # Step 1: Apply the mask to the original image (this keeps the fruit area, sets background to black)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Step 2: Convert the image to grayscale for easier pixel value comparison
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Step 3: Detect low pixel values below the threshold
    low_pixel_mask = gray_masked_image < threshold  # Pixels below threshold will be True
    high_pixel_mask = gray_masked_image > threshold
    
    # Step 4: Convert the result back to an image for visualization
    low_pixel_image = np.zeros_like(image)  # Create an empty image to highlight low pixel values
    low_pixel_image[low_pixel_mask] = [255, 255, 255]  # Set low-value pixels to red (for visualization)
    
    
    dark_pixel_count = np.count_nonzero(low_pixel_mask)/10000  # Number of dark pixels (low pixel values)
    # print(f"Number of dark pixels: {dark_pixel_count}")
    high_pixel_count = np.count_nonzero(high_pixel_mask)/1000
    # print(f"Number of high pixels: {high_pixel_count}")
    # Step 5: Display the results
    # cv2.imshow("Original Image with Low Pixel Values", low_pixel_image)  # Show the highlighted low pixel values
    # cv2.imshow("Masked Image", masked_image)  # Show the masked image
    # cv2.waitKey(0)  # Wait for a key press to close the images
    # cv2.destroyAllWindows()  # Close all OpenCV windows

    return high_pixel_count  # Return the mask of low pixel values

