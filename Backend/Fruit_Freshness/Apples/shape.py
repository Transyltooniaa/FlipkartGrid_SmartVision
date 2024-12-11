# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:15:10 2024

This code calculates the edge density of an image in blocks. It uses Canny edge detection to identify edges and calculates
the percentage of edge pixels in each block. The results can be used to analyze features such as wrinkles on fruit surfaces.

@author: jishu
"""

import cv2
import numpy as np

def calculate_edge_density(image, block_size=(50, 50)):
    """
    Calculates edge density in blocks for a given image.

    Parameters:
        image_path (str): Path to the input image.
        block_size (tuple): Dimensions of the blocks (height, width).

    Returns:
        list: Edge densities for each block (in percentage).
    """
    
    if image is None:
        print("Error: Image not found!")
        return None

    # Step 2: Resize the image for consistent processing
    resized_image = cv2.resize(image, (512, 512))

    # Step 3: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Step 4: Perform Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=100)

    # Step 5: Display the edge-detected image using OpenCV
    # cv2.imshow("Edge Detection", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 6: Calculate edge density for each block
    densities = []
    height, width = edges.shape
    block_height, block_width = block_size

    # Loop through the image in blocks
    for y in range(0, height, block_height):
        for x in range(0, width, block_width):
            # Define the block region
            block = edges[y:y+block_height, x:x+block_width]
            
            # Handle the case where the block goes out of image bounds
            block = block[:block_height, :block_width]
            
            # Calculate edge density
            edge_pixels = np.sum(block == 255)  # White pixels in the Canny output
            total_pixels = block.size
            density = (edge_pixels / total_pixels) * 100  # Convert to percentage
            densities.append(density)

    return np.mean(np.array(densities))


