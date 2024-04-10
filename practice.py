import cv2 as cv
# cv.imwrite(img1)으로 이미지 저장가능
# def func1() :
#     img1 = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)

#     if img1 is None:
#         print('failed!')
#         return
    
#     print('type(img1) : ', type(img1))
#     print('img1.shape:', img1.shape)
    
#     if len(img1.shape) == 2:
#         print('img1 is a grayscale img')
#     elif len(img1.shape) == 3:
#         print('img1 is a truecolor image')

#     cv.imshow('img1', img1)
#     cv.waitKey()
#     cv.destroyAllWindows()
# func1()


# def func3():
#     img1 = cv.imread('cat.png')
#     img2 = img1
#     img3 = img1.copy()

#     img1[:, :] = (0, 255, 255) #yellow

#     cv.imshow('img1', img1)
#     cv.imshow('img2', img2)
#     cv.imshow('img3', img3)
#     cv.waitKey()
#     cv.destroyAllWindows()

# func3()

# def func4():
#     img1 = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)
#     img2 = img1[200:400, 200:400]
#     img3 = img1[200:400, 200:400].copy()

#     img2 += 20

#     cv.imshow('img1', img1)
#     cv.imshow('img2', img2)
#     cv.imshow('img3', img3)
#     cv.waitKey()
#     cv.destroyAllWindows()

# func4()

# def func5() :
#     img1 = cv.imread('cat.png')
#     img2 = cv.bitwise_not(img1)
  
#     cv.imshow('img1', img1)
#     cv.imshow('img2', img2)

#     cv.waitKey()
#     cv.destroyAllWindows()

# func5()

# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print('cam open failed!')
#     exit()

# print('frame width : ', int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
# print('frame height:', int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
# print('frame count:', int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
# while True:
#     ret, frame = cap.read()
#     inversed = ~frame
#     if not ret:
#         break

#     cv.imshow('frame', frame)
#     cv.imshow('inversed', inversed)
#     if cv.waitKey(10) == 27:
#         break
# cv.destroyAllWindows()

################################

# cap = cv.VideoCapture('kanade.mp4')

# if not cap.isOpened():
#     print('video open failed!')
#     exit()
# print('frame width : ', int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
# print('frame height:', int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
# print('frame count:', int(cap.get(cv.CAP_PROP_FRAME_COUNT)))

# fps = cap.get(cv.CAP_PROP_FPS)
# print('fps:', fps)
# delay = round(1000/fps)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break
#     inversed = ~frame
#     cv.imshow('frame', frame)
#     cv.imshow('inversed', inversed)

#     if cv.waitKey(delay) == 27:
#         break
# cv.destroyAllWindows()

#################3

# cap = cv.VideoCapture('kanade.mp4')

# if not cap.isOpened():
#     print('failed')
#     exit()

# w = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv.CAP_PROP_FPS)
# delay = round(1000/fps)
# cc = cv.VideoWriter_fourcc(*'mp4v') #밖에서 안보이는 함수...?
# outputVideo = cv.VideoWriter('ouput.mp4', cc, fps, (w, h))

# if not outputVideo.isOpened():
#     print('file open failed')
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     inversed = ~frame
#     outputVideo.write(inversed)
#     cv.imshow('frame', frame)
#     cv.imshow('inversed', inversed)


#     if cv.waitKey(delay) == 27:
#         break
# cv.destroyAllWindows()

#############################

# import numpy as np

# img = np.full((400, 400, 3), 255, np.uint8)



# cv.line(img, (50, 50), (200, 50), (0, 0, 255))

# cv.line(img, (250, 50), (350, 100), (0, 0, 255), 1, cv.LINE_4) #상하좌우 네방향
# cv.line(img, (250, 70), (350, 120), (255, 0, 255), 1, cv.LINE_8) #대각선 방향으로도
# cv.line(img, (250, 90), (350, 140), (255, 0, 0), 1, cv.LINE_AA) #안티 엘리어싱

# cv.imshow('img', img)
# cv.waitKey()
# cv.destroyAllWindows()

##############################

# import numpy as np
# img = np.full((400, 400, 3), 255, np.uint8)

# cv.rectangle(img, (50, 50), (150, 100), (0, 0, 255), 2) #시작위치 x, y, rgb 그리고 두꼐
# cv.rectangle(img, (50, 150), (150, 200), (0, 0, 128), -1) #내부 채우기

# cv.circle(img, (300, 120), 30, (255, 255, 0), -1, cv.LINE_AA) #중심, 반지름, 색, 선두꼐(채우기), 라인타입
# cv.circle(img, (300, 120), 50, (255, 0, 0), 3, cv.LINE_AA)

# cv.ellipse(img, (120, 300), (60, 30), 20, 0, 270, (255, 255, 0), cv.FILLED, cv.LINE_AA) #cv.filled = -1
# cv.ellipse(img, (120, 300), (100, 50), 20, 0, 360, (0, 255, 0), 2, cv.LINE_AA)

# pts = np.array([[250, 250], [300, 250], [300, 300], [350, 300], [350, 350], [250, 350]])
# cv.polylines(img, [pts], True, (255, 0, 255), 2) #img, 점들, True면 도형 닫혀잇음-> 선하나 더, 선색, 두께

# cv.imshow('img', img)
# cv.waitKey()
# cv.destroyAllWindows()

###############################

# import numpy as np

# img = np.full((500, 800, 3), 255, np.uint8)

# cv.putText(img, 'FONT_HERSHEY_SIMPLEX', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
# #img, 글자, 시작위치, 폰트, 폰트크기, 라인타입, 색깔
# cv.imshow('img', img)
# cv.waitKey()
# cv.destroyAllWindows()

#####EVENTS

# img = cv.imread('cat.png')
# if img is None:
#     print('image load failed')
#     exit()

# cv.namedWindow('img')
# cv.imshow('img', img)

# while True:
#     keycode = cv.waitKey()
#     print(keycode)
#     if keycode is ord('i') or keycode is ord('I'):
#         img = ~img
#         cv.imshow('img', img)
#     elif keycode == 27 or keycode == ord('q') or keycode == ord('Q') :
#         break
# cv.destroyAllWindows()

##############################

# def on_mouse(event, x, y, flags, param):
#     global oldx, oldy

#     if event == cv.EVENT_LBUTTONDOWN:
#         oldx, oldy = x,y
#         print('event_lbuttondown: %d, %d' %(x, y))
#     elif event == cv.EVENT_LBUTTONUP:
#         print('event_lbuttonup: %d, %d' %(x, y))

#     elif event == cv.EVENT_MOUSEMOVE:
#         if flags & cv.EVENT_FLAG_LBUTTON:
#             cv.line(img, (oldx, oldy), (x, y), (0, 255, 255), 2)
#             cv.imshow('img', img)
#             oldx, oldy = x, y

# img = cv.imread('cat.png')
# if img is None:

#     exit()
# cv.namedWindow('img')
# cv.setMouseCallback('img', on_mouse)
# cv.imshow('img', img)
# cv.waitKey()
# cv.destroyAllWindows()

##################################

# import numpy as np

# def saturated(value):
#     if value>255:
#         value = 255
#     elif value < 0:
#         value = 0
#     return value
# def on_level_change(pos):
#     img[:] = saturated(pos* 16)
#     cv.imshow('image', img)
# img = np.zeros((400, 400), np.uint8)

# cv.namedWindow('image')
# cv.createTrackbar('level', 'image', 0, 16, on_level_change)

# cv.imshow('image', img)
# cv.waitKey()
# cv.destroyAllWindows()
#getTrackbarPos(), setTrackbarPos

# import numpy as np

# filename = 'data.json'

# def writeData():
#     name = 'Jane'
#     age = 10
#     pt1 = (100, 200)
#     scores = (80, 90, 50)
#     mat1 = np.array([[1.0, 1.5], [2.0, 3.2]], dtype = np.float32)
#     fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

#     if not fs.isOpened():
#         print('file open failed')
#         return
#     fs.write('name', name)
#     fs.write('age', age)
#     fs.write('point', pt1)
#     fs.write('scores', scores)
#     fs.write('data', mat1)

#     fs.release()
# def readData():
#     fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)

#     if not fs.isOpened():
#         print('file open failed')
#         return
#     name = fs.getNode('name').string()
#     age = int(fs.getNode('age').real())
#     pt1 = tuple(fs.getNode('point').mat().astype(np.int32).flatten())
#     scores = tuple(fs.getNode('scores').mat().flatten())
#     mat1 = fs.getNode('data').mat()

#     fs.release()

#     print('name: ', name)
#     print('age:', age)
#     print('point:', pt1)
#     print('scores:', scores)
#     print('data:')
#     print(mat1)
# writeData()
# readData()

# def mask_setTo():
#     src = cv.imread('cat.png', cv.IMREAD_COLOR)
#     mask = cv.imread('mask_smile.png', cv.IMREAD_GRAYSCALE)

#     if src is None or mask is None:
#         print('failed load')
#         return
#     src[mask>0] = (0, 255, 255)
#     #dst[mask>0] = src[mask > 0]

#     cv.imshow('rc', src)
#     cv.imshow('mask', mask)
#     cv.waitKey()
#     cv.destroyAllWindows()

# import numpy as np
# src = cv.imread('cat.png')

# if src is None:
#     exit()

# dst = np.empty(src.shape, dtype = src.dtype)
# tm = cv.TickMeter()
# tm.start()

# for y in range(src.shape[0]) :
#     for x in range(src.shape[1]) :
#         dst[y,x] = 255-src[y, x]
# tm.stop()
# print('took %4.3f ns.' % tm.getTimeMilli())

# cv.imshow('src', src)
# cv.imshow('dst', dst)
# cv.waitKey()
# cv.destroyAllWindows()


# img = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)
# minval, maxval, minpos, maxpos = cv.minMaxLoc(img)
# print(minval)
# print(maxval)
# print(minpos)
# print(maxpos)

# #normalize() 정규화 , 원소값 범위 특정하게 정규화
# import numpy as np

# src = np.array([[-1, -0.5, 0, 0.5, 1]], dtype = np.float32)
# dst = cv.normalize(src, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
# #cv.NORM_L1 -> manhattan dist cv.NORM_L2 -> euclidean
# print('src: ', src)
# print('dst: ', dst)

# #python round() 주의! 반올림 아님!!!! 2.5인 경우 가까운 짝수인 2로!
# print(round(2.5)) #0.5인경우 가장 가까운 짝수로, 아닌경우 그냥 반올림

import numpy as np


def brightness1():
    src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)
    if src is None:
        return
    
    dst = cv.add(src, 100)

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()
# brightness1()

#직접 더하면 255가 넘어가서 이상해짐 16진수 0x100 -> 0x00만되서 검어짐
def brightness4():
    src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)

    if src is None:
        return
    
    def update(pos):
        dst = cv.add(src, pos - 100)
        cv.imshow('dst', dst)
    cv.namedWindow('dst')
    cv.createTrackbar('Brightness', 'dst', 0, 200, update)
    update(0)

    cv.waitKey()
    cv.destroyAllWindows()
# brightness4()
def contrast1():
    src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)

    if src is None:
        return
    s = 2.0
    dst = cv.multiply(src, s)

    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()
# contrast1()
def contrast2():
    src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)

    if src is None:
        return
    alpha = 1.0
    dst = cv.convertScaleAbs(src, alpha=1+alpha, beta = -128*alpha)#y = (1 + alpha)(x - 128) + 128 -> (1 + alpha)x - 128alpha
    cv.imshow('src', src)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()
# contrast2()
def calcGrayHist(img):
    channels = [0]
    histsize = [256]
    histrange = [0, 256]
    hist = cv.calcHist([img], channels, None, histsize, histrange)

    return hist
def getGrayHistImage(hist):
    _, histMax, _, _ = cv.minMaxLoc(hist)
    imgHist = np.ones((100, 256), np.uint8) * 255
    for x in range(imgHist.shape[1]):
        pt1 = (x, 100)
        pt2 = (x, 100-int(hist[x, 0] * 100 / histMax))
        cv.line(imgHist, pt1, pt2, 0)

    return imgHist

# src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)
# hist = calcGrayHist(src)
# hist_img = getGrayHistImage(hist)

# cv.imshow('src', src)
# cv.imshow('srthist', hist_img)

# cv.waitKey()
# cv.destroyAllWindows()

import matplotlib.pyplot as plt
def histogram_stretching():
    src = cv.imread('cat.png', cv.IMREAD_COLOR)

    if src is None:
        return
    src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(src_ycrcb)
    oldy = y.copy()
    grayhist = getGrayHistImage(calcGrayHist(y))
    gmin, gmax, _, _ = cv.minMaxLoc(y)
    y = cv.convertScaleAbs(y, alpha = 255.0/(gmax - gmin), beta = -gmin * 255.0/(gmax - gmin))

    dst_ycrcb = cv.merge([y, cr, cb])
    dst = cv.cvtColor(dst_ycrcb, cv.COLOR_YCrCb2BGR)

    # cv.imshow('dst', dst)
    cv.imshow('oldgray', oldy)
    cv.imshow('dstgray', y)
    # cv.imshow('src', src)
    cv.imshow('srchist', grayhist)

    # cv.imshow('dst', dst)
    cv.imshow('dsthist', getGrayHistImage(calcGrayHist(y)))

    cv.imwrite('stretched_cat.png', dst)
    cv.waitKey()
    cv.destroyAllWindows()

# histogram_stretching()

def histogram_equalization() :
    src = cv.imread('cat.png')
    if src is None:
        return
    src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(src_ycrcb)

    oldy = y.copy()
    y = cv.equalizeHist(y)
    dst_ycrcb = cv.merge([y, cr, cb])

    dst = cv.cvtColor(dst_ycrcb, cv.COLOR_YCrCb2BGR)
    cv.imshow('src', src)
    cv.imshow('srchist', getGrayHistImage(calcGrayHist(oldy)))

    cv.imshow('dst', dst)
    cv.imshow('dsthist', getGrayHistImage(calcGrayHist(y)))
    channels = cv.split(dst)
    colors = ('b', 'g', 'r')
    for ch, color in zip(channels, colors):
        hist = cv.calcHist([ch], [0], None, [256], [0, 256])
        plt.plot(hist, color = color)
    plt.show()
    cv.imwrite('equalized_cat.png', dst)

    # cv.imshow('dsthsit', getGrayHistImage(calcGrayHist(dst)))

    cv.waitKey()
    cv.destroyAllWindows()
# histogram_equalization()

# src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)

# if src is None:
#     exit()

# emboss = np.array([[-1, -1, 0], 
#                    [-1, 0, 1],
#                    [0, 1, 1]], np.float32)
# dst = cv.filter2D(src, -1, emboss, delta = 128)

# cv.imshow('src', src)
# cv.imshow('dst', dst)

# cv.waitKey()
# cv.destroyAllWindows() 
#emboss

def blurring_mean():
    src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)
    if src is None:
        return
    cv.imshow('src', src)

    for ksize in range(3, 9, 2):
        dst = cv.blur(src, (ksize, ksize))

        desc = 'Mean: %dx%d' % (ksize, ksize)
        cv.putText(dst, desc, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv.LINE_AA)

        cv.imshow('dst', dst)
        cv.waitKey()

    cv.destroyAllWindows()
# blurring_mean()

# def blurring_gaussian():
#     src = cv.imread('cat.png', cv.IMREAD_GRAYSCALE)

#     if src is None:
#         return
#     cv.imshow('src', src)

#     for sigma in range(1,6)

###########7
src = cv.imread('test.png')
a = src.shape[0]
b = src.shape[1]

arr = [[1, 0, 0], [0, 1, 0]]
mat = np.array(arr).astype(np.float32)
dst = cv.warpAffine(src, mat, (0, 0))
cv.imshow('src', src)
cv.imshow('dst', dst)
cv.waitKey()

#1
nums = np.array([[[1, 4, 2], [5, 6, 2]], [[7, 5, 3], [8, 2, 9]]])
print(nums.shape)
arr = nums.swapaxes(0, 2)
print(nums)
print(arr)
print(arr.shape, arr[1, 1, 0])

# #2

# img = cv.imread('a.png', cv.IMREAD_GRAYSCALE)

# ori = img.copy()
# # img = ~img
# img[:, round(img.shape[1] / 2):] = ~img[:, round(img.shape[1] / 2):] #??????????????? shorter?
# cv.imshow('ori', ori)
# cv.imshow('result', img)
# cv.waitKey()
# cv.destroyAllWindows()

############## #4
# a = cv.imread('a.png', cv.IMREAD_COLOR)
# b = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)
# c = cv.imread('c.png', cv.IMREAD_COLOR)

# print(a.shape)
# print(b.shape)
# # a[b == 255] = (0, 0, 255); r = a  ##; is used
# # r = cv.copyTo(np.full((a.shape[0], a.shape[1], 3), (0, 0, 255), dtype = np.uint8), b, a)#src mask dst 순
# r = cv.copyTo(a, b, c)
# # a[b == 0] = c[b == 0]; r = a  ##ans4 -2
# # r, a[b==255] = a, (0, 0, 255)

# cv.imshow('result', r)

# cv.waitKey()
# cv.destroyAllWindows()

###############3

# a = np.zeros((120, 120), dtype = np.uint8)
# # a = [[3, 1, 2, 1], [1, 3, 1, 3], [1, 2, 0, 1], [3, 0, 3, 1]]
# a[0:30, 0: 30] = 3
# a[0:30, 30:60] = 1
# a[0:30, 60:90] = 2
# a[0:30, 90:] = 1
# a[30:60, 0 : 30] = 1
# a[30:60, 30:60] = 3
# a[30:60, 60:90] = 1
# a[30:60, 90:] = 3


# a[60:90, 0 : 30] = 1
# a[60:90, 30:60] = 2
# a[60:90, 60:90] = 0
# a[60:90, 90:] = 1

# a[90:, 0 : 30] = 3
# a[90:, 30:60] = 0
# a[90:, 60:90] = 3
# a[90:, 90:120] = 1
# # a = a * 64
# cv.imshow('hello', a)

# cv.waitKey()

# gmin, gmax, _, _ = cv.minMaxLoc(a)

# stret = cv.convertScaleAbs(a, alpha = 255.0 / (gmax - gmin), beta = -gmin * 255.0/(gmax - gmin))

# eqa = cv.equalizeHist(a)

# cv.imshow('stret', stret)
# cv.imshow('equ', eqa)

# cv.waitKey()
