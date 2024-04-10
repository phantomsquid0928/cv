import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations
import sys

argv = sys.argv
#for houghlines
def find_intersections(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            # Check if the lines are not too parallel by checking if their angle is not within 10 degrees
            if np.abs(theta1 - theta2) > np.deg2rad(10):
                A = np.array([
                    [np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]
                ])
                b = np.array([[rho1], [rho2]])
                # Find the intersection point
                intersection = np.linalg.solve(A, b)
                if np.all(np.isfinite(intersection)):
                    intersections.append((intersection[0][0], intersection[1][0]))
    return intersections

def filter_extreme_points(points, img_dim):
    return [point for point in points if 0 <= point[0] < img_dim[1] and 0 <= point[1] < img_dim[0]]
#for houghp
def segment_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return x, y
#for houghp
def find_intersectionsP(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            intersect = segment_intersection(line1, line2)
            if intersect is not None:
                intersections.append(intersect)
    return intersections

def show_with_matplotlib(color_img, title):
    img_RGB = color_img[:, :, ::-1]  
    plt.figure(figsize=(10, 10))  
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()

def extend_line(x1, y1, x2, y2, length):
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    dx = (x2 - x1) / line_length
    dy = (y2 - y1) / line_length
    
    x1_ext = int(x1 - dx * length)
    y1_ext = int(y1 - dy * length)
    x2_ext = int(x2 + dx * length)
    y2_ext = int(y2 + dy * length)
    
    return (x1_ext, y1_ext, x2_ext, y2_ext) 

def extend_lines(lines, length):
    extended_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        extended_lines.append([extend_line(x1, y1, x2, y2, length)])
    return extended_lines

def remove_similar_lines(lines, angle_threshold, distance_threshold):
    unique_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        rho = np.abs((x2 - x1) * y1 - (y2 - y1) * x1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        is_similar = False
        for unique_line in unique_lines:
            ux1, uy1, ux2, uy2 = unique_line[0]
            unique_angle = np.arctan2(uy2 - uy1, ux2 - ux1)
            unique_rho = np.abs((ux2 - ux1) * uy1 - (uy2 - uy1) * ux1) / np.sqrt((ux2 - ux1)**2 + (uy2 - uy1)**2)
            
            angle_difference = np.abs(angle - unique_angle)
            rho_difference = np.abs(rho - unique_rho)
            
            if angle_difference < np.deg2rad(angle_threshold) and rho_difference < distance_threshold:
                is_similar = True
                break
        
        if not is_similar:
            unique_lines.append(line)

    return unique_lines

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    distance = np.linalg.norm(rightMost - tl, axis=1)
    (br, tr) = rightMost[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

def largest_area_rect_from_hull(hull_points):
    max_area = 0
    best_quad = None

    for quad in combinations(hull_points, 4):
        quad = np.array(quad)
        x = quad[:, 0]
        y = quad[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area > max_area:
            max_area = area
            best_quad = quad

    if best_quad is not None:
        ordered_quad = order_points(best_quad)
        return ordered_quad
    else:
        return None

def warp_perspective(image, src_points, width, height):
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

def line_rect_intersection(line, img_width, img_height):
    x1, y1, x2, y2 = line[0]
    points = []

    borders = [
        ((0, 0), (img_width, 0)),  
        ((img_width, 0), (img_width, img_height)),  
        ((img_width, img_height), (0, img_height)),  
        ((0, img_height), (0, 0))  
    ]

    for border in borders:
        x3, y3 = border[0]
        x4, y4 = border[1]

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if det != 0:
            a = (x1 * y2 - y1 * x2)
            b = (x3 * y4 - y3 * x4)
            x = (a * (x3 - x4) - (x1 - x2) * b) / det
            y = (a * (y3 - y4) - (y1 - y2) * b) / det

            if (min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4) and
                    min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)):
                points.append((x, y))

    return points

def trim_line_to_image(lines, max_width, max_height):
    trimmed_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points = line_rect_intersection([(x1, y1, x2, y2)], max_width, max_height)
        if len(points) == 2:
            x1_trim, y1_trim = points[0]
            x2_trim, y2_trim = points[1]
            trimmed_lines.append([(int(x1_trim), int(y1_trim), int(x2_trim), int(y2_trim))])
    return trimmed_lines

def classify_and_filter_major_lines(lines, angle_tolerance=5):       #angle tolerance 만큼 같은 그룹으로 묶고 삭제
    if lines is None:
        return [], []

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180  
            if angle > 90:
                angle -= 180  

            if -angle_tolerance <= angle <= angle_tolerance or 180 - angle_tolerance <= angle <= 180:
                horizontal_lines.append((line, angle))
            elif 90 - angle_tolerance <= angle <= 90 + angle_tolerance or -90 - angle_tolerance <= angle <= -90 + angle_tolerance:
                vertical_lines.append((line, angle))

    def find_major_lines(classified_lines):
        if not classified_lines:
            return []

        classified_lines.sort(key=lambda x: x[1])

        angle_groups = {}
        for line, angle in classified_lines:
            rounded_angle = angle_tolerance * round(angle / angle_tolerance)
            if rounded_angle in angle_groups:
                angle_groups[rounded_angle].append(line)
            else:
                angle_groups[rounded_angle] = [line]

        major_group_angle = max(angle_groups, key=lambda k: len(angle_groups[k]))
        major_lines = angle_groups[major_group_angle]

        return major_lines

    major_horizontal_lines = find_major_lines(horizontal_lines)
    major_vertical_lines = find_major_lines(vertical_lines)

    return major_horizontal_lines, major_vertical_lines

#ycrcb cvt
#y, cr, cb = cv2.cvtColor(...)
#y->gaussan (0, 0) 1
#y->medianblur(9)

sharpener = np.array([[0, -1, 0],
                    [-1,  5, -1],
                    [0, -1, 0]])

filename = argv[1]
print(filename)
files = ['checker1.jpg',   #cannot find main color...
          'checker2.jpg', #fake 10x10 cor  thresh group 50, fake 100 cor
          'checker3.jpg',   #cor thresh gourp9 ok
          'checker4.png',   #cor  `` 9 TRUE OK
          'checker5.jpg',   #110 ?????????? need heuristic
            'checker6.png'] #150
img = cv2.imread(filename)
if img is None:
    print('filename is invalid')
    exit()
bil = cv2.bilateralFilter(img, -1, 1, 1) #1 1
img = cv2.addWeighted(img, 3, bil, -2, 0)

# ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# y, cr, cb = cv2.split(ycrcb_img)
# newy = cv2.equalizeHist(y)                   #ycrcb 한거까지 3개 비교 해보자.
# res = cv2.merge([newy, cr, cb])
# img = cv2.cvtColor(res, cv2.COLOR_YCrCb2BGR)

img_height = img.shape[0]
img_width = img.shape[1]

# print(int(np.sqrt(img_height*img_width) / 30)) 

top_border_width = 10   
bottom_border_width = 10
left_border_width = 10   
right_border_width = 10  


img = cv2.copyMakeBorder(img, top_border_width, bottom_border_width,
                                     left_border_width, right_border_width,
                                     cv2.BORDER_REPLICATE) 

img_height = img.shape[0]
img_width = img.shape[1]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray) ####################
# gray = cv2.equalizeHist(gray)
# _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
# show_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "test")
# Detect edges using Canny
# edges = cv2.Canny(binary, 50, 150, apertureSize=3)

v = np.mean(gray)

sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(gray, lower, upper)

# edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)


# lines = cv2.HoughLines(edges, 1, np.pi / 180, 130)
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=110, minLineLength=100, maxLineGap=100) #150 150 100int(np.sqrt(img_height * img_width) / 16)
# linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=60, maxLineGap=100) #150 150 100int(np.sqrt(img_height * img_width) / 16)


# show_with_matplotlib(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Grayscale')
# show_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 'Binary')
# show_with_matplotlib(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 'Edges')

if linesP is not None:
    img_with_linesP = img.copy()
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_linesP, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # show_with_matplotlib(img_with_linesP, 'Probabilistic Hough Lines')

extended_lines = extend_lines(linesP, 100000)
img_with_extended = img.copy()
for line in extended_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_with_extended, (x1, y1), (x2, y2), (0, 255, 0), 2)

# show_with_matplotlib(img_with_extended, 'Probabilistic Hough Lines - ext')

trimmed_line = trim_line_to_image(extended_lines, img_width, img_height)

unique_lines = remove_similar_lines(trimmed_line, angle_threshold=50, distance_threshold=np.sqrt(img_height * img_width) / 100) #10 10 #100

img_with_unique_lines = img.copy()
for line in unique_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_with_unique_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)

# show_with_matplotlib(img_with_unique_lines, 'Unique Lines')

intersections = find_intersectionsP(unique_lines)
# cv2.circle(img_with_extended, (i for i in intersections), 2, (0, 255, 255), 3)

# show_with_matplotlib(img_with_extended, 'Probabilistic Hough Lines - ext, circles')

intersections = filter_extreme_points(intersections, img.shape)

angle_threshold = 5  # Angle threshold in degrees
distance_threshold = 10  # Distance threshold in pixels

for points in intersections:
    cv2.circle(img_with_extended, (int(points[0]), int(points[1])), 5, (0, 0, 255), -1)
# show_with_matplotlib(img_with_extended, 'dots')
warped_image = None
if intersections:
    hull_points = cv2.convexHull(np.array(intersections, dtype=np.float32))
    hull_points = np.squeeze(hull_points)
    # print(hull_points)

    img_with_hull = img.copy()
    # draw the convex hull
    for point in hull_points:
        cv2.circle(img_with_hull, tuple(int(val) for val in point), 5, (0, 255, 0), -1)

    # Draw lines between the hull points
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the loop
    for i in range(len(hull_points) - 1):
        cv2.line(img_with_hull, tuple(int(val) for val in hull_points[i]), tuple(int(val) for val in hull_points[i+1]), (255, 0, 0), 2)

    # show_with_matplotlib(img_with_hull, 'Convex Hull')

    # hull_points = ...

    largest_quad = largest_area_rect_from_hull(hull_points)
    # print(largest_quad)
    if largest_quad is not None:
        
        max_width = 1000      #change to 1000x1000 in force -> checkboard is square so its ok
        max_height = 1000

        warped_image_gray = warp_perspective(edges, largest_quad, max_width, max_height)
        warped_image = warp_perspective(img, largest_quad, max_width, max_height)
        warped_image_withedge = warp_perspective(img_with_unique_lines, largest_quad, max_width, max_height)
        warped_image_binary = warp_perspective(binary, largest_quad, max_width, max_height)
        # show_with_matplotlib(warped_image, 'Warped Image')

        pass
    else:
        print("Could not find answer.") 
show_with_matplotlib(warped_image, 'RESULT')