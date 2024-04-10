import cv2
import matplotlib.pyplot as plt
import numpy as np

files = ['checker1.jpg',   #cannot find main color
          'checker2.jpg', #fake 10x10
          'checker3.jpg',   #cor
          'checker4.png',   #cor
          'checker5.jpg',   #110
            'checker6.png'] #150
img = cv2.imread(files[1])
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imggray, 130, 255, type=cv2.THRESH_BINARY)
cany = cv2.Canny(thresh, 50, 150)
contours, _ = cv2.findContours(cany, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # rect = cv2.minAreaRect(img)
    # box = cv2.boxPoints(rect)
    if len(cnt) <= 5 : continue
    ellipse = cv2.fitEllipse(cnt)
    
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    # cv2.drawContours(img, ellipse, 0, (0, 0, 255), 2)
plt.figure(figsize=(15, 15))
plt.imshow(img)
plt.show()