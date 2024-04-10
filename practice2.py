import cv2
import numpy as np

import matplotlib.pyplot as plt

def show_with_matplotlib(color_img, title):
    """Displays an image using Matplotlib."""
    img_RGB = color_img[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()

##########3 sobel filter
def sobel_derivative() :
    """sobel filter, -1, 2,-1    y axis deriv
                      0, 0, 0
                      1, 2, 1      
                      
                      -1, 0, 1
                      -2, 0, 2   x zxis deriv
                      -1, 0, 1    """
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    mx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype = np.float32)
    my = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype = np.float32)

    dx = cv2.filter2D(src ,-1, mx, delta = 128)
    dy = cv2.filter2D(src, -1, my, delta=128)

    cv2.imshow('src', src)
    cv2.imshow('dx', dx)
    cv2.imshow('dy', dy)
    cv2.waitKey()
    cv2.destroyAllWindows()

# sobel_derivative()
########################
def sobel_edge():
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    dx = cv2.Sobel(src, cv2.CV_32F, 1, 0) #src, ddepth, dx, dy
    dy = cv2.Sobel(src, cv2.CV_32F, 0, 1)
    #cv2.Scharr(src, cv2.cv_32f, 1, 0) 으로 해도 동일함

    fmag = cv2.magnitude(dx, dy)
    mag = np.uint8(np.clip(fmag, 0, 255))
    _, edge = cv2.threshold(mag, 150, 255, cv2.THRESH_BINARY)

    cv2.imshow('src', src)
    cv2.imshow('mag', mag)
    cv2.imshow('edge', edge)
    cv2.waitKey()
    cv2.destroyAllWindows()

# sobel_edge()
#cv2.Canny()
#  가우시안 -> 그라디언트계산->
#  비최대 억제-> 
# 이중임계갑슬 이용한 히스테리시스 에지 트래킹
#4단계 구성

def canny_edge():
    src = cv2.imread('cat.png' ,cv2.IMREAD_GRAYSCALE)

    dst1 = cv2.Canny(src, 50, 100)
    dst2 = cv2.Canny(src, 50, 150)
    
    show_with_matplotlib(cv2.cvtColor(dst1, cv2.COLOR_GRAY2BGR), 'f')
    show_with_matplotlib(cv2.cvtColor(dst2, cv2.COLOR_GRAY2BGR), 'dst2')

# canny_edge()

################
#HoughLines(), HoughLinesP()
import math
def hough_lines():
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLines(edge, 1, math.pi/180, 250)

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]) :
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            x0, y0 = rho * cos_t, rho * sin_t
            alpha = 1000
            pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
            pt2 = (int(x0 + alpha *sin_t), int(y0 - alpha * cos_t))
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    show_with_matplotlib(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), 'src')
    show_with_matplotlib(dst, 'dst')

# hough_lines()

def hough_line_segments():
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLinesP(edge,  1, np.pi/180, 160, minLineLength = 50, maxLineGap=5)

    dst  = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    print(lines.shape)
    if lines is not None:
        for i in lines:
            pt1 = (i[0][0], i[0][1])
            pt2 = (i[0][2], i[0][3])
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    show_with_matplotlib(dst, 'dst')
# hough_line_segments()

def hough_circles():
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    blurred = cv2.blur(src, (3, 3))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=30)
    #50보다 작으면 검출x param1 <- canny edge 높은임계값, param2 축적배열원소 30보다 크면 선택ㅡ 허프 그라디언트참조

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        for i in range(circles.shape[1]):
            # pr
            cx, cy, radius = circles[0][i]

            cv2.circle(dst, (cx, cy), radius, (0, 0, 255), 2, cv2.LINE_AA)

    show_with_matplotlib(dst, 'ff')

# hough_circles()

def hist_eq() :
    src = cv2.imread('cat.png')

    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(src_ycrcb)

    yy = cv2.equalizeHist(y)
    dst_ycrcb = cv2.merge([yy, cr, cb])

    dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

    show_with_matplotlib(dst, 'ff')
# hist_eq()

def on_hue_change(_=None) :
    lower_hue = cv2.getTrackbarPos('lower hue', 'mask')
    upper_hue = cv2.getTrackbarPos('Upper hue', 'mask')

    lowerb = (lower_hue, 100, 0)
    upperb = (upper_hue, 255, 255)

    mask = cv2.inRange(src_hsv, lowerb, upperb)

    cv2.imshow('mask', mask)
def hue():
    global src_hsv

    src = cv2.imread('cat.png')

    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    cv2.imshow('src', src)

    cv2.namedWindow('mask')
    cv2.createTrackbar('lower hue', 'mask', 40, 179,on_hue_change)
    cv2.createTrackbar('Upper hue', 'mask', 80, 179, on_hue_change)

    on_hue_change(0)

    cv2.waitKey()
    cv2.destroyAllWindows()
# hue()

def histback():
    ref = cv2.imread('cat.png')
    mask = cv2.imread('mask.bmp', cv2.IMREAD_GRAYSCALE)
    ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

    ranges = [0, 256] + [0, 256]
    hist = cv2.calcHist([ref_ycrcb], [1, 2], mask, [128, 128], ranges, 1)


def binarization() :
    filename = 'cat.png'
    src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    def on_threshold(pos):
        _, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('dst', dst)
    # cv2.imshow('src', src)
    cv2.namedWindow('dst')
    cv2.createTrackbar('Threshold', 'dst', 0, 255, on_threshold)
    cv2.setTrackbarPos('Threshold', 'dst', 128)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

binarization()
def adaptive_thresh() :
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
    def on_trackbar(pos):
        bsize = pos 
        if bsize % 2 == 0: bsize = bsize-1
        if bsize < 3: bsize = 3

        dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, 5)

        cv2.imshow('dst', dst)
    cv2.imshow('src', src)

    cv2.namedWindow('dst')
    cv2.createTrackbar('Block size', 'dst', 0, 200, on_trackbar)
    cv2.setTrackbarPos('Block size', 'dst', 11)

    cv2.waitKey()
    cv2.destroyAllWindows()

# adaptive_thresh()
def erode_dilate():
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst1 = cv2.erode(src_bin, None)
    dst2 = cv2.dilate(src_bin, None)

    plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
    plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
    plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('dilate')
    plt.show()


# erode_dilate()

def open_close() :
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY)

    dst1 = cv2.morphologyEx(src_bin, cv2.MORPH_OPEN, None)
    dst2 = cv2.morphologyEx(src_bin, cv2.MORPH_CLOSE, None)

    plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
    plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
    plt.subplot(223), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('open')
    plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('close')

    plt.show()

# open_close()

def labeling_basic():
    src = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 1, 0],
                    [1, 1, 1, 1, 0, 0 ,0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    
    src = src * 255

    cnt, labels = cv2.connectedComponents(src)

    print(labels)
    print(cnt) #실제 개수는 cnt-1
# labeling_basic()

#cnt, labels, stats, centroids = cv2.connectedComponentWithStats(src_bin) src_bin = cv2.threshold(src, 0, 255, cv2.thresh_binary|thresh_otsu)

#(x, y, w, h, area) = stats[i]
#pt1 = (x, y)
#pt2 = (x + w, y + h)
#cv2.rectangle(dst, pt1, pt2, (0, 255, 255))

import random
def contours_basic():
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dst, contours, -1, (0, 255, 0), 2)
    # for contour in contours:
    #     cv2.drawContours(dst, [contour], -1, (0, 255, 0), 2)
    show_with_matplotlib(dst, 'ff')
# contours_basic()

def contours_hier() :
    src = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.Canny(src, 50, 150)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    idx = 0
    while idx >= 0:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(dst, contours, idx, c, -1, cv2.LINE_8, hierarchy)
        idx = hierarchy[0, idx, 0]

    show_with_matplotlib(dst, 'src')

# contours_hier()
def contourff() :
    img = cv2.imread('cat.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours :
        if cv2.contourArea(contour) < 400:
            continue
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)

        vtc = len(approx)
        if vtc == 3: #tri
            print('r')
        elif vtc == 4:
            print('ff')
        else :
            length = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            ratio = 4. * math.pi * area / (length * length)

            if ratio > 0.85:
                print('fffffffdssfs')
    show_with_matplotlib(img, 'img')
    cv2.waitKey()
    cv2.destroyAllWindows()

# contourff()


def template_matcher() :
    img = cv2.imread('building1.jpg')
    temp = cv2.imread('building1tmp.jpg')

    noise = np.zeros(img.shape, np.int32)
    cv2.randn(noise, 0, 10)
    img = cv2.add(img, noise, dtype = cv2.CV_8UC3)
    res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)

    (th, tw) = temp.shape[:2]
    cv2.rectangle(img, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)

    show_with_matplotlib(temp, 'f')
    show_with_matplotlib(cv2.cvtColor(res_norm, cv2.COLOR_GRAY2BGR), 'ff')
    show_with_matplotlib(img, 'img')

# template_matcher()

def detect_face():
    src = cv2.imread('cat.png')

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = classifier.detectMultiScale(src)

    for(x, y, w, h) in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    show_with_matplotlib(src, 'f')
detect_face()
def hog_dis() :
    src = cv2.imread('')
