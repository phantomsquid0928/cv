import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

argv = sys.argv
img = cv2.imread(argv[1])
if img is None:
    print('invalid filename')
    exit()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('CLICK')
plt.axis('image')

points = plt.ginput(4)
plt.close()

pts1 = np.array(points, dtype="float32")

width, height = 300, 300
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

result = cv2.warpPerspective(img, matrix, (width, height))

result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.imshow(result_rgb)
plt.axis('off')
plt.show()
