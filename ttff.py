import numpy as np
import cv2
import matplotlib.pyplot as plt

img_height = 100
img_width = 100
# We need to find red and blue colors with the same Y value in YCrCb color space
# Define a blue and a red color in RGB
blue_rgb = np.array([0, 0, 255], dtype=np.uint8)
red_rgb = np.array([255, 0, 0], dtype=np.uint8)

# Convert them to YCrCb
blue_ycrcb = cv2.cvtColor(np.uint8([[blue_rgb]]), cv2.COLOR_RGB2YCrCb)[0][0]
red_ycrcb = cv2.cvtColor(np.uint8([[red_rgb]]), cv2.COLOR_RGB2YCrCb)[0][0]

# Adjust Y values to be the same
average_y = (blue_ycrcb[0] + red_ycrcb[0]) // 2
blue_ycrcb[0] = average_y
red_ycrcb[0] = average_y

# Create an image with left blue and right red in YCrCb space
ycrcb_img_adjusted = np.zeros((img_height, img_width, 3), dtype=np.uint8)
ycrcb_img_adjusted[:, :img_width//2] = blue_ycrcb
ycrcb_img_adjusted[:, img_width//2:] = red_ycrcb

# Convert YCrCb image to RGB for display
ycrcb_img_rgb_adjusted = cv2.cvtColor(ycrcb_img_adjusted, cv2.COLOR_YCrCb2RGB)

# Convert to grayscale
gray_img_ycrcb_adjusted = cv2.cvtColor(ycrcb_img_rgb_adjusted, cv2.COLOR_RGB2GRAY)

# Apply Canny edge detection
edges_ycrcb_adjusted = cv2.Canny(gray_img_ycrcb_adjusted, 100, 200)

# Display the images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(ycrcb_img_rgb_adjusted, cv2.COLOR_RGB2BGR))
axs[0].set_title('Adjusted Original Image (YCrCb)')
axs[0].axis('off')

axs[1].imshow(gray_img_ycrcb_adjusted, cmap='gray')
axs[1].set_title('Adjusted Grayscale Image (YCrCb)')
axs[1].axis('off')

axs[2].imshow(edges_ycrcb_adjusted, cmap='gray')
axs[2].set_title('Adjusted Edge Detection (YCrCb)')
axs[2].axis('off')

plt.show()
