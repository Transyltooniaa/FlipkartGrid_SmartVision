# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:38:54 2024

@author: jishu
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_orange_yellow_frequency(image):
    
    if image is None:
        print("Error: Image not found!")
        return None

    # Step 2: Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Step 3: Extract the Hue channel
    hue_channel = hsv_image[:, :, 0]
    
    # Step 4: Calculate the histogram for the Hue channel
    hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
    
    # Step 5: Calculate the frequency for orange and yellow hues, excluding red
    orange_range = range(10, 30)  # Hue values for orange
    yellow_range = range(30, 60)  # Hue values for yellow
    
    # Sum the frequencies for the orange range
    orange_frequency = sum(hist[hue] for hue in orange_range)
    yellow_frequency = sum(hist[hue] for hue in yellow_range)
    
    # Combine frequencies for orange and yellow
    total_orange_yellow_frequency = orange_frequency + yellow_frequency

    # Normalize the histogram for percentage calculation
    total_pixels = hue_channel.size
    orange_percentage = (orange_frequency / total_pixels) * 100
    yellow_percentage = (yellow_frequency / total_pixels) * 100
    combined_percentage = (total_orange_yellow_frequency / total_pixels) * 100
    
    return (orange_percentage, yellow_percentage)

    # print(f"Orange Percentage: {orange_percentage[0]:.2f}%")
    # print(f"Yellow Percentage: {yellow_percentage[0]:.2f}%")
    # print(f"Combined Percentage of Orange and Yellow: {combined_percentage[0]:.2f}%")

    # Step 6: Visualize the histogram using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.title("Hue Histogram (Ignoring Red)")
    # plt.xlabel("Hue Value")
    # plt.ylabel("Frequency")
    # plt.plot(hist, color='orange', label='Hue Histogram')
    # plt.axvspan(10, 30, color='orange', alpha=0.3, label='Orange Range')
    # plt.axvspan(30, 60, color='yellow', alpha=0.3, label='Yellow Range')
    # plt.legend()
    # plt.show()

    # # Step 7: Display the image using OpenCV
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
