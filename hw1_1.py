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
        print("Could not find a quadrilateral.") 
        
#TODO: 각도가 너무 다른 선 거르기
#TODO: img 크기 1000x1000으로 고정됫으니까 나머지 hough 벼눗값도 조정

# res = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
linesp2 = cv2.HoughLinesP(warped_image_gray, 1, np.pi/180, threshold= 140, minLineLength = 90, maxLineGap= 90)

linesp2 = extend_lines(linesp2, 2000)
linesp2 = trim_line_to_image(linesp2, 1000, 1000)
linesp2 = remove_similar_lines(linesp2, 10, 70)

hor, ver = classify_and_filter_major_lines(linesp2)
filtered_lines = hor+ver

# show_with_matplotlib(cv2.cvtColor(warped_image_gray, cv2.COLOR_GRAY2BGR), 'hello')
# show_with_matplotlib(warped_image_withedge, "with original edge")
warped_edges_col = cv2.cvtColor(warped_image_gray, cv2.COLOR_GRAY2BGR)

# _, contours, _ = cv2.findContours()
# circles = cv2.HoughCircles(warped_edges, cv2.HOUGH_GRADIENT, 1, 200, param1 = 250, param2 = 10, minRadius = 10, maxRadius= 200)
for lines in linesp2:
    x1, y1, x2, y2 = lines[0]
    cv2.line(warped_edges_col, (x1, y1), (x2, y2), (0, 0, 255), 2)

warped_image_afteredge = warped_image.copy()
for lines in filtered_lines:
    x1, y1, x2, y2 = lines[0]
    cv2.line(warped_edges_col, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.line(warped_image_afteredge, (x1, y1), (x2, y2), (0, 255, 0), 2)
# for cir in circles[0]:
#     cv2.circle(warped_edges_col, (int(cir[0]), int(cir[1])), int(cir[2]), (0, 0, 255), 5)
# show_with_matplotlib(warped_edges_col, 'helo')



def count_color_changes_across_lines(image):
    
    lower_green = np.array([0, 250, 0])
    upper_green = np.array([0, 255, 0])
    mask = cv2.inRange(image, lower_green, upper_green)
    # Find contours of the green lines
    # show_with_matplotlib(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 'mask')
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(~mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.drawContours(image,z contours, -1, (0, 0, 255), 3)
    # show_with_matplotlib(image, 'contour works?')
    cnt_groups = {}
    count = 0
    alpha = 25
    medw = np.median([cv2.boundingRect(cnt)[2] for cnt in contours])
    medh = np.median([cv2.boundingRect(cnt)[3] for cnt in contours])

    # print(medw, medh)

    target = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(w, h)
        if (medw - alpha <= w and medw + alpha >= w) :
            if (medh - alpha <= h and medh + alpha >=h) :
                target.append(cnt.copy())
    # print(target)
    cv2.drawContours(image, target, -1, (255, 0, 0), 3)
    # show_with_matplotlib(image, 'contour?')
    return target, medw, medh



alpha = 20
t1, medw1, medh1= count_color_changes_across_lines(warped_image_afteredge.copy())
t2, medw2, medh2= count_color_changes_across_lines(warped_image_withedge.copy())
score1 = 0
score2 = 0
if abs(medw1 - medh1) < alpha:
    score1 += 1
if abs(medw2 - medh2) < alpha:
    score2 += 1
if medw1 > 60 and medh1 > 60:
    score1 += 1
if medw2 > 60 and medh2 > 60:
    score2 += 1
if len(t1) > len(t2):
    score1 += 1
else :
    score2 += 1
if score1 > score2 :
    t = t1
    # print('t1 selected')
else :
    t = t2
    # print('t2 selected')

def sort_contours(contours, board_size, image_size):
    sorted_contours = []
    expected_distance = image_size[0] // board_size  # Assuming square image and square board

    # Calculate the centroid of each contour
    centroids = []
    for contour in contours:
        M = cv2.moments(contour) #need refector
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
        else:
            centroids.append((0, 0))  # Adding a dummy value for contours with zero area

    # Sort the centroids based on their expected position
    for i in range(board_size):
        for j in range(board_size):
            expected_position = (j * expected_distance + expected_distance // 2, 
                                 i * expected_distance + expected_distance // 2)
            # Find the index of the closest centroid
            closest_index = np.argmin([np.linalg.norm(np.array(expected_position) - np.array(centroid)) for centroid in centroids])
            sorted_contours.append(contours[closest_index])
            # Remove the used centroid and contour
            del centroids[closest_index]
            del contours[closest_index]

    return sorted_contours

if len(t) > 81 :
    pseudo_size = 10
else :
    pseudo_size = 8
# print("PSEUDO-" + str(pseudo_size))
t = sort_contours(t, pseudo_size, (1000,1000))

warped_image_contoures = warped_image.copy()

cv2.drawContours(warped_image_contoures, t, -1, (255, 0, 0), 2)
# show_with_matplotlib(warped_image_contoures, 'result')
board_color_array = np.zeros((pseudo_size, pseudo_size), dtype=object)

def get_median_color(roi, mask, exclude_color=(0, 255, 0)):
    contour_extracted = roi[mask == 255]

    reshaped = contour_extracted.reshape(-1, 3)

    filtered_colors = reshaped[~np.all(reshaped == (0, 0, 0), axis=1)]
    filtered_colors = filtered_colors[~np.all(filtered_colors == exclude_color, axis=1)]

    if filtered_colors.size == 0:
        return None

    mean_color = np.median(filtered_colors, axis=0)

    return mean_color.astype(int)  
def get_mean_color(roi, mask, exclude_color=(0, 255, 0)):
    contour_extracted = roi[mask == 255]

    reshaped = contour_extracted.reshape(-1, 3)

    filtered_colors = reshaped[~np.all(reshaped == (0, 0, 0), axis=1)]
    filtered_colors = filtered_colors[~np.all(filtered_colors == exclude_color, axis=1)]

    if filtered_colors.size == 0:
        return None

    mean_color = np.mean(filtered_colors, axis=0)

    return mean_color.astype(int)  


for i, cnt in enumerate(t):
    x, y, w, h = cv2.boundingRect(cnt) #if x y is almost mid, get color : save color changes as array : if color != central color and bounding circle
    # warped_image[x + int(w/2) #color pick, if color is not fit : piece, no : find curve in this area -> sth detected : piece no : nopiece.
    center_x = x + w//2
    center_y = y + h//2
    mask = np.zeros((1000, 1000), dtype = np.uint8)
    
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    # show_with_matplotlib(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 'mask')
    # res = cv2.copyTo(warped_image, mask)

    part = warped_image[y:y+h, x:x+w]
    maskpiece = mask[y:y+h, x:x+w]
    
    contour_extracted = np.zeros_like(part)
    contour_extracted[maskpiece == 255] = part[maskpiece == 255]

    # extracted_lab = cv2.cvtColor(contour_extracted, cv2.COLOR_BGR2LAB)
    extracted_lab = contour_extracted
    
    major_color = get_mean_color(extracted_lab, maskpiece)
    # print(major_color)
    row = i // pseudo_size
    col = i % pseudo_size
    board_color_array[row, col] = tuple(major_color)

def are_colors_similar(color1, color2, threshold=100): #rgb
    delta_e = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    return delta_e < threshold

def are_colors_similar_bgr(color1, color2, threshold=10): #bgr
    delta_e = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    return delta_e < threshold



def group_similar_colors(color_pattern, threshold=50): ###bgr thresh 50
    groups = []
    for color in color_pattern:
        found = False
        for group in groups:
            if are_colors_similar_bgr(group[0], color, threshold):
                group.append(color)
                found = True
                break
        if not found:
            groups.append([color])
    return groups

def check_fake_board_edges(board_color_pattern, board_size, threshold=100): #bgr thresh : 100
    edge_colors = []
    for i in [0, board_size - 1]:
        edge_colors.extend([
            board_color_pattern[i, 0], 
            board_color_pattern[i, -1], 
            board_color_pattern[0, i], 
            board_color_pattern[-1, i]
        ])
    
    color_groups = group_similar_colors(edge_colors, threshold)

    if len(color_groups) == 1:
        return True
    return False


board_size = pseudo_size
if pseudo_size == 10 and check_fake_board_edges(board_color_array, pseudo_size):
    board_size = 8
    temp_board_color_array = np.zeros((board_size, board_size, 3), dtype=int)
    for i in range(1, 9):
        for j in range(1, 9):
            temp_board_color_array[i-1, j-1] = board_color_array[i, j]
    board_color_array = temp_board_color_array

print(f"The checkerboard size is: {board_size} x {board_size} ")
