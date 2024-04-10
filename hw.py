import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Load your image and preprocess
image_path = 'checker2.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 9, 75, 75)

img_height, img_width = image.shape[:2]
print(np.sqrt(img_height * img_width) / 30)
# 2. Edge detection using Canny
edges = cv2.Canny(blurred, 70, 150, apertureSize=3)

dilated_edges = cv2.dilate(edges, None, iterations=1)
# Display the result of Canny edge detection
plt.figure(figsize=(10, 10))
plt.imshow(dilated_edges, cmap='gray')
plt.title("Canny Edges")
plt.show()

# 3. Hough Line Transform
lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 150, minLineLength = np.sqrt(img_height * img_width) / 30, maxLineGap = np.sqrt(img_height * img_width) / 300)

if lines is None:
    print("No lines were detected.")
    exit()


def compute_angle(line):
    """Compute angle (in degrees) of the line with respect to x-axis."""
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180  # Convert to positive angles

def line_midpoint(line):
    """Compute the midpoint of a line."""
    x1, y1, x2, y2 = line
    return (x1 + x2) / 2, (y1 + y2) / 2

def merge_similar_lines(lines, delta_angle=40, delta_position=np.sqrt(img_height * img_width) / 25):
    """Merge lines with similar angles and positions."""
    if not lines:
        return []

    # Compute angles and midpoints for each line
    angles = [compute_angle(line) for line in lines]
    midpoints = [line_midpoint(line) for line in lines]

    lines_info = list(zip(lines, angles, midpoints))
    lines_info.sort(key=lambda x: (x[1], x[2]))  # Sort by angle then midpoint

    merged_lines = []
    current_group = [lines_info[0]]
    current_avg_angle = lines_info[0][1]
    current_avg_midpoint = lines_info[0][2]

    for line_info in lines_info[1:]:
        line, angle, midpoint = line_info

        angle_diff = abs(angle - current_avg_angle)
        midpoint_diff = np.linalg.norm(np.array(midpoint) - np.array(current_avg_midpoint))

        if angle_diff < delta_angle and midpoint_diff < delta_position:
            current_group.append(line_info)
            current_avg_angle = np.mean([x[1] for x in current_group])
            current_avg_midpoint = np.mean(np.array([x[2] for x in current_group]), axis=0)
        else:
            avg_line = tuple(np.mean([x[0] for x in current_group], axis=0).astype(int))
            merged_lines.append(avg_line)
            current_group = [line_info]
            current_avg_angle = angle
            current_avg_midpoint = midpoint

    # Handle the last group
    if current_group:
        avg_line = tuple(np.mean([x[0] for x in current_group], axis=0).astype(int))
        merged_lines.append(avg_line)

    return merged_lines


# 4. Draw the lines on a copy of the original image
lines_merged = merge_similar_lines([line[0] for line in lines])

line_image = image.copy()

for line in lines_merged:
    x1, y1, x2, y2 = line
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 5. Display the image with detected lines
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines using Hough Transform")
plt.show()

def extend_line(x1, y1, x2, y2, length=1000):
    # Normalize line direction
    line_dir = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    line_dir /= np.linalg.norm(line_dir)
    
    # Compute points along the line in both directions
    x1_new = int(x1 - length * line_dir[0])
    y1_new = int(y1 - length * line_dir[1])
    x2_new = int(x2 + length * line_dir[0])
    y2_new = int(y2 + length * line_dir[1])
    
    return x1_new, y1_new, x2_new, y2_new

def line_intersection(line1, line2):
    # Unpack lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Compute determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:  # lines are parallel
        return None
    
    # Compute intersection point
    px = int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det)
    py = int(((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det)
    
    return int(px), int(py)




def clip_line_to_image(x1, y1, x2, y2, img_width, img_height):
    # Direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        y1_clip = max(0, min(y1, y2))
        y2_clip = min(img_height - 1, max(y1, y2))
        return x1, y1_clip, x2, y2_clip

    # If the line is horizontal
    if dy == 0:
        x1_clip = max(0, min(x1, x2))
        x2_clip = min(img_width - 1, max(x1, x2))
        return x1_clip, y1, x2_clip, y2
    
    # Possible t values for intersections with image boundaries
    t_values = [(0 - x1) / dx,           # left edge
                (img_width - 1 - x1) / dx,  # right edge
                (0 - y1) / dy,           # top edge
                (img_height - 1 - y1) / dy]  # bottom edge
    
    # Filter t values to get valid intersections
    t_values = [t for t in t_values if 0 <= t <= 1]
    
    # If there's no valid intersection, the line is outside the image
    if not t_values:
        return None
    
    # Get the two extreme t values
    t_min, t_max = min(t_values), max(t_values)
    
    # Calculate clipped line endpoints using the extreme t values
    x1_clip = int(x1 + t_min * dx)
    y1_clip = int(y1 + t_min * dy)
    x2_clip = int(x1 + t_max * dx)
    y2_clip = int(y1 + t_max * dy)
    
    return x1_clip, y1_clip, x2_clip, y2_clip


# Extend lines and compute intersections

cropped_extended_lines = []
intersections = []



for line in lines_merged:
    x1, y1, x2, y2 = line
    extended = extend_line(x1, y1, x2, y2)
    cropped_extended = clip_line_to_image(extended[0], extended[1], extended[2], extended[3], img_width, img_height)
    if cropped_extended:
        cropped_extended_lines.append(cropped_extended)

for i in range(len(cropped_extended_lines)):
    for j in range(i+1, len(cropped_extended_lines)):
        intersection = line_intersection(cropped_extended_lines[i], cropped_extended_lines[j])
        if intersection:
            intersections.append(intersection)

# Draw extended lines and intersections
extended_line_img = image.copy()

for line in cropped_extended_lines:
    cv2.line(extended_line_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)  # Green lines
margin = 10

# Filter intersections within the expanded boundaries of the image

# from itertools import combinations
# import numpy as np

# def calculate_polygon_area(points):
#     # Assuming points is a NumPy array with shape (4, 2)
#     # This calculates the area using the shoelace formula
#     x = points[:, 0]
#     y = points[:, 1]
#     return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# def is_convex_quadrilateral(points):
#     # This is a non-trivial function. You need to implement the logic
#     # to check whether four points form a convex quadrilateral.
#     # This can be done by checking the cross product of adjacent vectors
#     # and ensuring they all have the same sign (all positive or all negative).
#     pass

# def compute_intersections(lines):
#     # This would be a function using the line_intersection method you already have
#     # to compute the intersections of all the lines
#     pass

# # Generate all combinations of four lines
# line_combinations = list(combinations(cropped_extended_lines, 4)) #cropped-> 2points x1y1 x2y2
# max_area = 0
# max_quad = None

# # Test each combination
# for lines in line_combinations:
#     # Compute all intersection points for these lines
#     intersection_points = compute_intersections(lines)
#     if len(intersection_points) == 4:
#         area = calculate_polygon_area(np.array(intersection_points))
#         if area > max_area:
#             max_area = area
#             max_quad = lines

# max_quad now holds the lines that form the largest quadrilateral


valid_intersections = [point for point in intersections 
                       if -margin <= point[0] <= img_width + margin and 
                          -margin <= point[1] <= img_height + margin]

if not valid_intersections:
    print("No intersections found!")
    exit()
for point in valid_intersections:
    cv2.circle(extended_line_img, point, 5, (0, 0, 255), -1)  # Red intersection points

# Display
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(extended_line_img, cv2.COLOR_BGR2RGB))
plt.title("Extended Lines and Intersections")
plt.show()
margin = 10

# Filter intersections within the expanded boundaries of the image


top_left = min(valid_intersections, key=lambda p: (p[0] + p[1]))
top_right = max(valid_intersections, key=lambda p: p[0] - p[1])
bottom_left = max(valid_intersections, key=lambda p: p[1] - p[0])
bottom_right = max(valid_intersections, key=lambda p: (p[0] + p[1]))

# Based on the outermost intersections, infer the outermost lines
leftmost_line = (top_left, bottom_left)
rightmost_line = (top_right, bottom_right)
topmost_line = (top_left, top_right)
bottommost_line = (bottom_left, bottom_right)

outermost_lines = [topmost_line, bottommost_line, leftmost_line, rightmost_line]

# Draw outermost lines on a copy of the original image
outermost_line_img = image.copy()

for line in outermost_lines:
    cv2.line(outermost_line_img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 255), 2)  # Yellow lines for outermost lines

# Display
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(outermost_line_img, cv2.COLOR_BGR2RGB))
plt.title("Outermost Lines of Checkerboard")
plt.show()
