#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2

img = cv2.imread("image.png") # Read image

# Setting parameter values
t_lower = 50 # Lower Threshold
t_upper = 150 # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)

cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[15]:


#Gradient Calculation
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('image.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.show()


# In[25]:


# Step 3: Non-maximum suppression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from scipy import ndimage

# Parameters to be tuned in different functions
LOW_THRESHOLD_RATIO = 0.09
HIGH_THRESHOLD_RATIO = 0.25
WEAK_PIXEL = 100
STRONG_PIXEL = 300

SAMPLE_IMAGE = "image.png"

# A few utility functions to preprocess and visualize the results
def rgb2gray(rgb):
    """
        Converts an RGB image into grayscale
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def visualize(imgs, m, n):
    """
        Visualize images with the matplotlib library
    """
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        plt_idx = i+1
        plt.subplot(m, n, plt_idx)
        plt.imshow(img, cmap='gray')
    plt.show()

image = mpimg.imread(SAMPLE_IMAGE)
gray_image = rgb2gray(image)
img_list = [image, gray_image]

visualize(img_list, 1, 2)
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M):
        for j in range(1, N):
            try:
                q = 300
                r = 300

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z


# In[35]:


#Double Threshold
import cv2
import numpy as np

# Load the image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Define your lower and upper threshold values
lower_threshold = 100
upper_threshold = 200

# Create masks based on the thresholds
lower_mask = image >= lower_threshold
upper_mask = image <= upper_threshold

# Initialize the result image with zeros
result_image = np.zeros_like(image)

# Assign different values to pixels based on the masks
result_image[lower_mask] = 255  # Foreground
result_image[upper_mask] = 127  # Intermediate

# Display the result
cv2.imshow('Double Threshold', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[36]:


#edge tracking of hysteresis
import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load the image.")
else:
    # Define your low and high threshold values
    low_threshold = 50
    high_threshold = 150

    # Apply Canny edge detection with the defined thresholds
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Display the result
    cv2.imshow('Edge Tracking by Hysteresis', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:




