# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# from itertools import combinations

# #for houghlines
# def find_intersections(lines):
#     intersections = []
#     for i, line1 in enumerate(lines):
#         for line2 in lines[i+1:]:
#             rho1, theta1 = line1[0]
#             rho2, theta2 = line2[0]
#             # Check if the lines are not too parallel by checking if their angle is not within 10 degrees
#             if np.abs(theta1 - theta2) > np.deg2rad(10):
#                 A = np.array([
#                     [np.cos(theta1), np.sin(theta1)],
#                     [np.cos(theta2), np.sin(theta2)]
#                 ])
#                 b = np.array([[rho1], [rho2]])
#                 # Find the intersection point
#                 intersection = np.linalg.solve(A, b)
#                 if np.all(np.isfinite(intersection)):
#                     intersections.append((intersection[0][0], intersection[1][0]))
#     return intersections

# def filter_extreme_points(points, img_dim):
#     # Filter points to keep only those that are within the image boundaries
#     return [point for point in points if 0 <= point[0] < img_dim[1] and 0 <= point[1] < img_dim[0]]
# #for houghp
# def segment_intersection(line1, line2):
#     """Finds the intersection of two lines given in segment form."""
#     x1, y1, x2, y2 = line1[0]
#     x3, y3, x4, y4 = line2[0]

#     # Line AB represented as a1x + b1y = c1
#     a1 = y2 - y1
#     b1 = x1 - x2
#     c1 = a1 * x1 + b1 * y1

#     # Line CD represented as a2x + b2y = c2
#     a2 = y4 - y3
#     b2 = x3 - x4
#     c2 = a2 * x3 + b2 * y3

#     determinant = a1 * b2 - a2 * b1

#     if determinant == 0:
#         # The lines are parallel; no intersection
#         return None
#     else:
#         # The lines are not parallel; find the intersection point
#         x = (b2 * c1 - b1 * c2) / determinant
#         y = (a1 * c2 - a2 * c1) / determinant
#         return x, y
# #for houghp
# def find_intersectionsP(lines):
#     intersections = []
#     for i, line1 in enumerate(lines):
#         for line2 in lines[i+1:]:
#             intersect = segment_intersection(line1, line2)
#             if intersect is not None:
#                 intersections.append(intersect)
#     return intersections

# def show_with_matplotlib(color_img, title):
#     """Helper function to display an image with Matplotlib."""
#     img_RGB = color_img[:, :, ::-1]  # Convert BGR to RGB
#     plt.figure(figsize=(10, 10))  # Set the figure size to 10x10 inches
#     plt.imshow(img_RGB)
#     plt.title(title)
#     plt.axis('off')  # Hide the axis
#     plt.show()

# def extend_line(x1, y1, x2, y2, length):
#     """Extend the line segment by a certain length."""
#     line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
#     # Normalize the direction vector (dx, dy) of the line segment
#     dx = (x2 - x1) / line_length
#     dy = (y2 - y1) / line_length
    
#     # Extend the line segment in both directions by 'length'
#     x1_ext = int(x1 - dx * length)
#     y1_ext = int(y1 - dy * length)
#     x2_ext = int(x2 + dx * length)
#     y2_ext = int(y2 + dy * length)
    
#     return (x1_ext, y1_ext, x2_ext, y2_ext) 

# def extend_lines(lines, length):
#     """Extend all line segments in 'lines' by 'length'."""
#     extended_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         extended_lines.append([extend_line(x1, y1, x2, y2, length)])
#     return extended_lines

# def remove_similar_lines(lines, angle_threshold, distance_threshold):
#     unique_lines = []

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         angle = np.arctan2(y2 - y1, x2 - x1)
#         rho = np.abs((x2 - x1) * y1 - (y2 - y1) * x1) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
#         # Check if line is similar to any unique line
#         is_similar = False
#         for unique_line in unique_lines:
#             ux1, uy1, ux2, uy2 = unique_line[0]
#             unique_angle = np.arctan2(uy2 - uy1, ux2 - ux1)
#             unique_rho = np.abs((ux2 - ux1) * uy1 - (uy2 - uy1) * ux1) / np.sqrt((ux2 - ux1)**2 + (uy2 - uy1)**2)
            
#             angle_difference = np.abs(angle - unique_angle)
#             rho_difference = np.abs(rho - unique_rho)
            
#             # If the line is similar by angle and distance, mark it as similar
#             if angle_difference < np.deg2rad(angle_threshold) and rho_difference < distance_threshold:
#                 is_similar = True
#                 break
        
#         # If the line is not similar to any line in unique_lines, add it to the list
#         if not is_similar:
#             unique_lines.append(line)

#     return unique_lines

# def order_points(pts):
#     # Sort the points based on their x-coordinates
#     xSorted = pts[np.argsort(pts[:, 0]), :]

#     # Grab the left-most and right-most points from the sorted
#     # x-coordinate points
#     leftMost = xSorted[:2, :]
#     rightMost = xSorted[2:, :]

#     # Now, sort the left-most coordinates according to their
#     # y-coordinates so we can grab the top-left and bottom-left points
#     leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
#     (tl, bl) = leftMost

#     # Now that we have the top-left coordinate, use it as an
#     # anchor to calculate the Euclidean distance between the
#     # top-left and right-most points; by the Pythagorean
#     # theorem, the point with the largest distance will be
#     # our bottom-right point
#     distance = np.linalg.norm(rightMost - tl, axis=1)
#     (br, tr) = rightMost[np.argsort(distance)[::-1], :]

#     # Return the coordinates in top-left, top-right,
#     # bottom-right, and bottom-left order
#     return np.array([tl, tr, br, bl], dtype="float32")

# def largest_area_rect_from_hull(hull_points):
#     max_area = 0
#     best_quad = None

#     # Generate all combinations of four points
#     for quad in combinations(hull_points, 4):
#         # Compute the area of the quadrilateral using the Shoelace formula
#         quad = np.array(quad)
#         x = quad[:, 0]
#         y = quad[:, 1]
#         area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
#         # Check if this quadrilateral has a larger area than the current max
#         if area > max_area:
#             max_area = area
#             best_quad = quad

#     # If we have found a quadrilateral
#     if best_quad is not None:
#         # Order the points in the quadrilateral
#         ordered_quad = order_points(best_quad)
#         return ordered_quad
#     else:
#         return None

# def warp_perspective(image, src_points, width, height):
#     # Define the destination points (the points where the corners will be after the perspective transform)
#     # We're assuming a rectangle here, so the top-left corner will go to (0,0) and the bottom-right to (width-1, height-1)
#     dst_points = np.array([
#         [0, 0],
#         [width - 1, 0],
#         [width - 1, height - 1],
#         [0, height - 1]
#     ], dtype='float32')

#     # Compute the perspective transform matrix
#     M = cv2.getPerspectiveTransform(src_points, dst_points)

#     # Perform the warp perspective
#     warped = cv2.warpPerspective(image, M, (width, height))

#     return warped

# def line_rect_intersection(line, img_width, img_height):
#     x1, y1, x2, y2 = line[0]
#     points = []

#     # Define the borders
#     borders = [
#         ((0, 0), (img_width, 0)),  # Top border
#         ((img_width, 0), (img_width, img_height)),  # Right border
#         ((img_width, img_height), (0, img_height)),  # Bottom border
#         ((0, img_height), (0, 0))  # Left border
#     ]

#     for border in borders:
#         # Each border is defined by two points (x3, y3), (x4, y4)
#         x3, y3 = border[0]
#         x4, y4 = border[1]

#         # Calculate the determinant
#         det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#         if det != 0:
#             # Lines are not parallel, calculate intersection
#             a = (x1 * y2 - y1 * x2)
#             b = (x3 * y4 - y3 * x4)
#             x = (a * (x3 - x4) - (x1 - x2) * b) / det
#             y = (a * (y3 - y4) - (y1 - y2) * b) / det

#             # Check if the intersection is within the border segment and the line segment
#             if (min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4) and
#                     min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)):
#                 points.append((x, y))

#     return points

# def trim_line_to_image(lines, max_width, max_height):
#     """Trim the lines to the image boundaries."""
#     trimmed_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         points = line_rect_intersection([(x1, y1, x2, y2)], max_width, max_height)
#         # Ensure two points are returned before creating a line
#         if len(points) == 2:
#             x1_trim, y1_trim = points[0]
#             x2_trim, y2_trim = points[1]
#             trimmed_lines.append([(int(x1_trim), int(y1_trim), int(x2_trim), int(y2_trim))])
#     return trimmed_lines

# def classify_and_filter_major_lines(lines, angle_tolerance=5):
#     """
#     Classify lines as horizontal or vertical, then within each classification,
#     keep only the major lines that share similar angles.

#     :param lines: Output from cv2.HoughLinesP.
#     :param angle_tolerance: The tolerance for considering angles to be the same, in degrees.
#     :return: Two lists of lines, one for horizontal and one for vertical major lines.
#     """
#     if lines is None:
#         return [], []

#     # Initialize lists to hold horizontal and vertical lines
#     horizontal_lines = []
#     vertical_lines = []

#     # Group lines by their angle
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180  # Angle in degrees between 0 and 180
#             if angle > 90:
#                 angle -= 180  # Adjust angles to be from -90 to 90

#             # Classify lines as horizontal or vertical based on angle
#             if -angle_tolerance <= angle <= angle_tolerance or 180 - angle_tolerance <= angle <= 180:
#                 horizontal_lines.append((line, angle))
#             elif 90 - angle_tolerance <= angle <= 90 + angle_tolerance or -90 - angle_tolerance <= angle <= -90 + angle_tolerance:
#                 vertical_lines.append((line, angle))

#     # Function to find the major lines from classified lines
#     def find_major_lines(classified_lines):
#         if not classified_lines:
#             return []

#         # Sort the lines by angle
#         classified_lines.sort(key=lambda x: x[1])

#         # Group lines by similar angle using a histogram-like approach
#         angle_groups = {}
#         for line, angle in classified_lines:
#             rounded_angle = angle_tolerance * round(angle / angle_tolerance)
#             if rounded_angle in angle_groups:
#                 angle_groups[rounded_angle].append(line)
#             else:
#                 angle_groups[rounded_angle] = [line]

#         # Find the largest group of lines
#         major_group_angle = max(angle_groups, key=lambda k: len(angle_groups[k]))
#         major_lines = angle_groups[major_group_angle]

#         return major_lines

#     # Find major lines for both horizontal and vertical lines
#     major_horizontal_lines = find_major_lines(horizontal_lines)
#     major_vertical_lines = find_major_lines(vertical_lines)

#     return major_horizontal_lines, major_vertical_lines

# #ycrcb cvt
# #y, cr, cb = cv2.cvtColor(...)
# #y->gaussan (0, 0) 1
# #y->medianblur(9)

# sharpener = np.array([[0, -1, 0],
#                     [-1,  5, -1],
#                     [0, -1, 0]])

# files = ['checker1.jpg',   #cannot find main color...
#           'checker2.jpg', #fake 10x10 cor  thresh group 50, fake 100 cor
#           'checker3.jpg',   #cor thresh gourp9 ok
#           'checker4.png',   #cor  `` 9 TRUE OK
#           'checker5.jpg',   #110 ?????????? need heuristic
#             'checker6.png'] #150
# img = cv2.imread(files[4])
# bil = cv2.bilateralFilter(img, -1, 1, 1) #1 1
# img = cv2.addWeighted(img, 3, bil, -2, 0)

# # ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# # y, cr, cb = cv2.split(ycrcb_img)
# # newy = cv2.equalizeHist(y)                   #ycrcb 한거까지 3개 비교 해보자.
# # res = cv2.merge([newy, cr, cb])
# # img = cv2.cvtColor(res, cv2.COLOR_YCrCb2BGR)

# img_height = img.shape[0]
# img_width = img.shape[1]

# print(int(np.sqrt(img_height*img_width) / 30)) 

# top_border_width = 10   # Add 50 pixels of padding on the top
# bottom_border_width = 10 # Add 50 pixels of padding on the bottom
# left_border_width = 10   # Add 50 pixels of padding on the left
# right_border_width = 10  # Add 50 pixels of padding on the right


# # Create a border around the image
# img = cv2.copyMakeBorder(img, top_border_width, bottom_border_width,
#                                      left_border_width, right_border_width,
#                                      cv2.BORDER_REPLICATE) # White border

# img_height = img.shape[0]
# img_width = img.shape[1]

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # gray = cv2.equalizeHist(gray) ####################
# # gray = cv2.equalizeHist(gray)
# # Binarize the grayscale image
# _, binary_piece = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)                #######thresh for finding piece
# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 11, 2)
# show_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "test")
# # Detect edges using Canny
# # edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# # Calculate the median pixel value
# v = np.mean(gray)

# # Apply automatic Canny edge detection using the computed median
# sigma = 0.33
# lower = int(max(0, (1.0 - sigma) * v))
# upper = int(min(255, (1.0 + sigma) * v))
# edges = cv2.Canny(gray, lower, upper)
# edges_piece = cv2.Canny(binary, lower, upper)
# # edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
# show_with_matplotlib(cv2.cvtColor(edges_piece, cv2.COLOR_GRAY2BGR), 'pieces?')


# # Detect lines using Hough Transform
# # lines = cv2.HoughLines(edges, 1, np.pi / 180, 130)
# linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=110, minLineLength=100, maxLineGap=100) #150 150 100int(np.sqrt(img_height * img_width) / 16)
# # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=60, maxLineGap=100) #150 150 100int(np.sqrt(img_height * img_width) / 16)


# show_with_matplotlib(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Grayscale')
# show_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 'Binary')
# show_with_matplotlib(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 'Edges')

# if linesP is not None:
#     img_with_linesP = img.copy()
#     for line in linesP:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(img_with_linesP, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     show_with_matplotlib(img_with_linesP, 'Probabilistic Hough Lines')

# extended_lines = extend_lines(linesP, 100000)
# img_with_extended = img.copy()
# for line in extended_lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img_with_extended, (x1, y1), (x2, y2), (0, 255, 0), 2)

# show_with_matplotlib(img_with_extended, 'Probabilistic Hough Lines - ext')

# trimmed_line = trim_line_to_image(extended_lines, img_width, img_height)

# unique_lines = remove_similar_lines(trimmed_line, angle_threshold=50, distance_threshold=np.sqrt(img_height * img_width) / 100) #10 10 #100

# img_with_unique_lines = img.copy()
# for line in unique_lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img_with_unique_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)

# # Now you can use the show_with_matplotlib function to display the image
# show_with_matplotlib(img_with_unique_lines, 'Unique Lines')

# intersections = find_intersectionsP(unique_lines)
# # cv2.circle(img_with_extended, (i for i in intersections), 2, (0, 255, 255), 3)

# # show_with_matplotlib(img_with_extended, 'Probabilistic Hough Lines - ext, circles')

# # Filter out points that are outside the image
# intersections = filter_extreme_points(intersections, img.shape)

# angle_threshold = 5  # Angle threshold in degrees
# distance_threshold = 10  # Distance threshold in pixels

# # Assuming 'intersections' is a list of intersection points and 'extended_lines' is a list of lines
# # filtered_intersections = filter_close_parallel_intersections(intersections, extended_lines, angle_threshold, distance_threshold)

# for points in intersections:
#     cv2.circle(img_with_extended, (int(points[0]), int(points[1])), 5, (0, 0, 255), -1)
# show_with_matplotlib(img_with_extended, 'dots')
# # Find the convex hull of the intersections
# warped_image = None
# if intersections:
#     hull_points = cv2.convexHull(np.array(intersections, dtype=np.float32))
#     hull_points = np.squeeze(hull_points)
#     # print(hull_points)

#     img_with_hull = img.copy()
#     # Draw the convex hull
#     for point in hull_points:
#         cv2.circle(img_with_hull, tuple(int(val) for val in point), 5, (0, 255, 0), -1)

#     # Draw lines between the hull points
#     hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the loop
#     for i in range(len(hull_points) - 1):
#         cv2.line(img_with_hull, tuple(int(val) for val in hull_points[i]), tuple(int(val) for val in hull_points[i+1]), (255, 0, 0), 2)

#     show_with_matplotlib(img_with_hull, 'Convex Hull')

#     # Assuming 'hull_points' is a Nx2 array of your convex hull points
#     # hull_points = ...

#     # Find the largest area quadrilateral from the convex hull points
#     largest_quad = largest_area_rect_from_hull(hull_points)
#     # print(largest_quad)
#     if largest_quad is not None:
#         # Now you can use largest_quad for perspective transformation
        
# # Assuming 'img' is the image you want to transform and 'largest_quad' is the array of points from the previous function
# # You also need to specify the width and height of the resulting image
# # For example, you could use the width and height of the original image or a specific size you want

# # Calculate the maximum width and height of the qua
#         max_width = 1000
#         max_height = 1000

#         # Now we can perform the warp perspective with the largest quad and the calculated width and height
#         warped_image_gray = warp_perspective(edges, largest_quad, max_width, max_height)
#         warped_image = warp_perspective(img, largest_quad, max_width, max_height)
#         warped_image_withedge = warp_perspective(img_with_unique_lines, largest_quad, max_width, max_height)
#         warped_image_binary = warp_perspective(binary, largest_quad, max_width, max_height)
#         # Show the result
#         show_with_matplotlib(warped_image, 'Warped Image')

#         pass
#     else:
#         print("Could not find a quadrilateral.") 
        
# #TODO: 각도가 너무 다른 선 거르기
# #TODO: img 크기 1000x1000으로 고정됫으니까 나머지 hough 벼눗값도 조정

# # res = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
# linesp2 = cv2.HoughLinesP(warped_image_gray, 1, np.pi/180, threshold= 140, minLineLength = 90, maxLineGap= 90)

# linesp2 = extend_lines(linesp2, 2000)
# linesp2 = trim_line_to_image(linesp2, 1000, 1000)
# linesp2 = remove_similar_lines(linesp2, 10, 70)

# hor, ver = classify_and_filter_major_lines(linesp2)
# filtered_lines = hor+ver

# show_with_matplotlib(cv2.cvtColor(warped_image_gray, cv2.COLOR_GRAY2BGR), 'hello')
# show_with_matplotlib(warped_image_withedge, "with original edge")
# warped_edges_col = cv2.cvtColor(warped_image_gray, cv2.COLOR_GRAY2BGR)

# # _, contours, _ = cv2.findContours()
# # circles = cv2.HoughCircles(warped_edges, cv2.HOUGH_GRADIENT, 1, 200, param1 = 250, param2 = 10, minRadius = 10, maxRadius= 200)
# for lines in linesp2:
#     x1, y1, x2, y2 = lines[0]
#     cv2.line(warped_edges_col, (x1, y1), (x2, y2), (0, 0, 255), 2)

# warped_image_afteredge = warped_image.copy()
# for lines in filtered_lines:
#     x1, y1, x2, y2 = lines[0]
#     cv2.line(warped_edges_col, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     cv2.line(warped_image_afteredge, (x1, y1), (x2, y2), (0, 255, 0), 2)
# # for cir in circles[0]:
# #     cv2.circle(warped_edges_col, (int(cir[0]), int(cir[1])), int(cir[2]), (0, 0, 255), 5)
# show_with_matplotlib(warped_edges_col, 'helo')



# def count_color_changes_across_lines(image):
    
#     lower_green = np.array([0, 250, 0])
#     upper_green = np.array([0, 255, 0])
#     mask = cv2.inRange(image, lower_green, upper_green)
#     # Find contours of the green lines
#     show_with_matplotlib(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 'mask')
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.dilate(mask, kernel, iterations=1)
#     contours, _ = cv2.findContours(~mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
#     # cv2.drawContours(image,z contours, -1, (0, 0, 255), 3)
#     # show_with_matplotlib(image, 'contour works?')
#     cnt_groups = {}
#     count = 0
#     alpha = 25
#     medw = np.median([cv2.boundingRect(cnt)[2] for cnt in contours])
#     medh = np.median([cv2.boundingRect(cnt)[3] for cnt in contours])

#     # print(medw, medh)

#     target = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         # print(w, h)
#         if (medw - alpha <= w and medw + alpha >= w) :
#             if (medh - alpha <= h and medh + alpha >=h) :
#                 target.append(cnt.copy())
#     # print(target)
#     cv2.drawContours(image, target, -1, (255, 0, 0), 3)
#     show_with_matplotlib(image, 'contour?')
#     return target, medw, medh



# alpha = 20
# t1, medw1, medh1= count_color_changes_across_lines(warped_image_afteredge.copy())
# t2, medw2, medh2= count_color_changes_across_lines(warped_image_withedge.copy())
# score1 = 0
# score2 = 0
# if abs(medw1 - medh1) < alpha:
#     score1 += 1
# if abs(medw2 - medh2) < alpha:
#     score2 += 1
# if medw1 > 60 and medh1 > 60:
#     score1 += 1
# if medw2 > 60 and medh2 > 60:
#     score2 += 1
# if len(t1) > len(t2):
#     score1 += 1
# else :
#     score2 += 1
# if score1 > score2 :
#     t = t1
#     print('t1 selected')
# else :
#     t = t2
#     print('t2 selected')
# print(f"The checkerboard size is:  ")

# # if len(t) < 80 :    #need tour mid, find change color <- cannot find correct answer on img 1
# #     print('its 8x8')
# # else :
# #     print('itx 10x10')
# # sorted(t, key=lambda x : cv2.boundingRect(x)[1] * 1000 + cv2.boundingRect(x)[0], reverse = True)
# def sort_contours(contours, board_size, image_size):
#     sorted_contours = []
#     expected_distance = image_size[0] // board_size  # Assuming square image and square board

#     # Calculate the centroid of each contour
#     centroids = []
#     for contour in contours:
#         M = cv2.moments(contour) #need refector
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             centroids.append((cX, cY))
#         else:
#             centroids.append((0, 0))  # Adding a dummy value for contours with zero area

#     # Sort the centroids based on their expected position
#     for i in range(board_size):
#         for j in range(board_size):
#             expected_position = (j * expected_distance + expected_distance // 2, 
#                                  i * expected_distance + expected_distance // 2)
#             # Find the index of the closest centroid
#             closest_index = np.argmin([np.linalg.norm(np.array(expected_position) - np.array(centroid)) for centroid in centroids])
#             sorted_contours.append(contours[closest_index])
#             # Remove the used centroid and contour
#             del centroids[closest_index]
#             del contours[closest_index]

#     return sorted_contours

# if len(t) > 81 :
#     pseudo_size = 10
# else :
#     pseudo_size = 8
# print("PSEUDO-" + str(pseudo_size))
# t = sort_contours(t, pseudo_size, (1000,1000))

# warped_image_contoures = warped_image.copy()

# cv2.drawContours(warped_image_contoures, t, -1, (255, 0, 0), 2)
# show_with_matplotlib(warped_image_contoures, 'result')
# board_color_array = np.zeros((pseudo_size, pseudo_size), dtype=object)


# # def get_major_color(roi, mask, exclude_color=(0, 255, 0)):
#     # # Extract the region of interest using the mask
#     # contour_extracted = roi[mask == 255]
#     # # Find the major color in the extracted contour
#     # colors, counts = np.unique(contour_extracted.reshape(-1, 3), axis=0, return_counts=True)
    
#     # # Sort colors by count in descending order
#     # sorted_indices = np.argsort(-counts)

#     # colors = colors[sorted_indices]
#     # counts = counts[sorted_indices]

#     # # Ignore black (background) color and the excluded color
#     # valid_indices = np.where((~np.all(colors == (0, 0, 0), axis=1)) & (~np.all(colors == exclude_color, axis=1)))

#     # # If no valid color is found, return a default color or raise an error
#     # if valid_indices[0].size == 0:
#     #     return None

#     # # Return the most common valid color
#     # major_color = colors[valid_indices][0]
#     # return major_color


# def get_median_color(roi, mask, exclude_color=(0, 255, 0)):
#     # Extract the region of interest using the mask
#     contour_extracted = roi[mask == 255]

#     # Reshape for easier manipulation
#     reshaped = contour_extracted.reshape(-1, 3)

#     # Filter out the excluded colors
#     filtered_colors = reshaped[~np.all(reshaped == (0, 0, 0), axis=1)]
#     filtered_colors = filtered_colors[~np.all(filtered_colors == exclude_color, axis=1)]

#     # If no valid color is found, return a default color or raise an error
#     if filtered_colors.size == 0:
#         return None

#     # Calculate the mean color
#     mean_color = np.median(filtered_colors, axis=0)

#     return mean_color.astype(int)  # Convert to integer for color representation
# def get_mean_color(roi, mask, exclude_color=(0, 255, 0)):
#     # Extract the region of interest using the mask
#     contour_extracted = roi[mask == 255]

#     # Reshape for easier manipulation
#     reshaped = contour_extracted.reshape(-1, 3)

#     # Filter out the excluded colors
#     filtered_colors = reshaped[~np.all(reshaped == (0, 0, 0), axis=1)]
#     filtered_colors = filtered_colors[~np.all(filtered_colors == exclude_color, axis=1)]

#     # If no valid color is found, return a default color or raise an error
#     if filtered_colors.size == 0:
#         return None

#     # Calculate the mean color
#     mean_color = np.mean(filtered_colors, axis=0)

#     return mean_color.astype(int)  # Convert to integer for color representation

# # Example usage
# # roi = ...  # Region of interest from your image
# # mask = ...  # Mask defining the ROI
# # mean_color = get_mean_color(roi, mask)


# for i, cnt in enumerate(t):
#     x, y, w, h = cv2.boundingRect(cnt) #if x y is almost mid, get color : save color changes as array : if color != central color and bounding circle
#     # warped_image[x + int(w/2) #color pick, if color is not fit : piece, no : find curve in this area -> sth detected : piece no : nopiece.
#     center_x = x + w//2
#     center_y = y + h//2
#     mask = np.zeros((1000, 1000), dtype = np.uint8)
    
#     cv2.drawContours(mask, [cnt], -1, 255, -1)
#     # show_with_matplotlib(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 'mask')
#     # res = cv2.copyTo(warped_image, mask)

#     part = warped_image[y:y+h, x:x+w]
#     maskpiece = mask[y:y+h, x:x+w]
    
#     # Create a new image to place the extracted contour
#     contour_extracted = np.zeros_like(part)
#     contour_extracted[maskpiece == 255] = part[maskpiece == 255]

#     # extracted_lab = cv2.cvtColor(contour_extracted, cv2.COLOR_BGR2LAB)
#     extracted_lab = contour_extracted
#     # show_with_matplotlib(contour_extracted, 'piece' + str(i))
#     # colors, counts = np.unique(contour_extracted.reshape(-1, 3), axis=0, return_counts=True)
#     # # Ignore black (background) color
#     # non_black_indices = np.where(~np.all(colors == 0, axis=1))
#     # major_color = colors[non_black_indices][counts[non_black_indices].argmax()]
    
#     major_color = get_mean_color(extracted_lab, maskpiece)
#     # print(major_color)
#     # Save the major color to the board color array
#     row = i // pseudo_size
#     col = i % pseudo_size
#     board_color_array[row, col] = tuple(major_color)
    
#     # masked_area = warped_image[mask == 255]
#     # show_with_matplotlib(contour_extracted, 'masked ' + str(i))
#     # Draw the index number of the contour on the image
#     # cv2.putText(warped_image, str(i), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
#     #             0.5, (255, 255, 255), 2)


# def are_colors_similar(color1, color2, threshold=300): #bgr
#     # Calculate the Euclidean distance between the two colors in RGB space
#     delta_e = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
#     return delta_e < threshold

# def are_colors_similar_bgr(color1, color2, threshold=10): #bgr
#     # Calculate the Euclidean distance between the two colors in BGR space
#     delta_e = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
#     return delta_e < threshold



# # Function to group similar colors
# def group_similar_colors(color_pattern, threshold=50): ###bgr thresh 50
#     groups = []
#     for color in color_pattern:
#         found = False
#         for group in groups:
#             if are_colors_similar_bgr(group[0], color, threshold):
#                 group.append(color)
#                 found = True
#                 break
#         if not found:
#             groups.append([color])
#     return groups




# def check_fake_board_edges(board_color_pattern, board_size, threshold=100): #bgr thresh : 100
#     # Check if the first and last rows and columns have the same color
#     edge_colors = []
#     for i in [0, board_size - 1]:
#         edge_colors.extend([
#             board_color_pattern[i, 0], 
#             board_color_pattern[i, -1], 
#             board_color_pattern[0, i], 
#             board_color_pattern[-1, i]
#         ])
    
#     # Group similar colors
#     color_groups = group_similar_colors(edge_colors, threshold)

#     # If there's only one group of similar colors, it's likely a fake edge
#     if len(color_groups) == 1:
#         return True
#     return False

# # Assuming you have the are_colors_similar and group_similar_colors functions defined as above

# # Initialize the board color pattern

# board_size = pseudo_size
# if pseudo_size == 10 and check_fake_board_edges(board_color_array, pseudo_size):
#     # Adjust the board size to 8x8
#     board_size = 8
#     print('FAKE')
#     # Create a new array without the fake edges
#     temp_board_color_array = np.zeros((board_size, board_size, 3), dtype=int)
#     for i in range(1, 9):
#         for j in range(1, 9):
#             temp_board_color_array[i-1, j-1] = board_color_array[i, j]
#     board_color_array = temp_board_color_array

# print(board_color_array.shape)

# # Assuming board_size is the actual size of the board (8 or 10)
# # Assuming pseudo_size is the size of the board_color_array which may be larger due to fake edges

# # Initialize the board color pattern with the actual board size
# board_color_pattern = np.zeros((board_size, board_size, 3), dtype=int)

# # Process the middle rows and columns based on the actual board size
# # middle_indices = [3, 4] if board_size == 8 else [4, 5]

# # # Assuming board_color_array is already filled with the major colors in Lab space
# # # ...

# # square_size = 50
# # # checkerboard_image_forcheck = np.zeros((board_size * square_size, board_size * square_size, 3), dtype=np.uint8)


# # # Now, when you check the color pattern for the middle rows and columns:
# # for axis in [0, 1]:  # 0 for rows, 1 for columns
# #     for idx in middle_indices:
# #         color_pattern = []
# #         for i in range(board_size):
# #             # Calculate the index for the board_color_array
# #             contour_idx = idx * board_size + i if axis == 0 else i * board_size + idx
# #             # Adjust index if pseudo_size is larger (fake edges present)
# #             # if pseudo_size > board_size:
# #             #     contour_idx += (pseudo_size - board_size) // 2 * (pseudo_size if axis == 0 else 1)
# #             major_color = board_color_array[contour_idx // board_size][contour_idx % board_size]
# #             color_pattern.append(major_color)

# #             # cv2.rectangle(checkerboard_image_forcheck, (0, 0), (50, 50), tuple(int(c) for c in major_color), -1)

# #             # show_with_matplotlib(checkerboard_image_forcheck, 'checkin')
        
# #         # Group similar colors
# #         color_groups = group_similar_colors(color_pattern)
        
# #         # Check if there are exactly two color groups and each appears at least board_size // 2 times
# #         if len(color_groups) == 2 and all(len(group) >= board_size // 2 for group in color_groups):
# #             # Found a valid color pattern, predict the entire board
# #             for i in range(board_size):
# #                 for j in range(board_size):
# #                     if (i + j) % 2 == 0:
# #                         board_color_pattern[i, j] = color_groups[0][0]  # Assuming the first color in the group is representative
# #                     else:
# #                         board_color_pattern[i, j] = color_groups[1][0]  # Assuming the first color in the group is representative
# #             break





# # def is_checkerboard_pattern(color_groups, board_size):
# #     # Ensure there are exactly two color groups
# #     if len(color_groups) != 2:
# #         return False
    
# #     # Ensure each color group appears in an alternating pattern
# #     first_color = color_groups[0][0]
# #     second_color = color_groups[1][0]
# #     expected_pattern = [first_color if i % 2 == 0 else second_color for i in range(board_size)]
    
# #     for group in color_groups:
# #         # Check if the group follows the expected pattern
# #         if not all(are_colors_similar(color, expected_color) for color, expected_color in zip(group, expected_pattern)):
# #             return False
    
# #     return True

# # middle_indices = [3, 4] if board_size == 8 else [4, 5]

# # for axis in [0, 1]:  # 0 for rows, 1 for columns
# #     for idx in middle_indices:
# #         color_pattern = []
# #         for i in range(board_size):
# #             # Calculate the index for the board_color_array
# #             contour_idx = idx * board_size + i if axis == 0 else i * board_size + idx
# #             major_color = board_color_array[contour_idx // board_size][contour_idx % board_size]
# #             color_pattern.append(major_color)
        
# #         # Group similar colors
# #         color_groups = group_similar_colors(color_pattern)
        
# #         # Check if the color pattern forms a checkerboard pattern
# #         if is_checkerboard_pattern(color_groups, board_size):
# #             # Found a valid color pattern, predict the entire board
# #             for i in range(board_size):
# #                 for j in range(board_size):
# #                     if (i + j) % 2 == 0:
# #                         board_color_pattern[i, j] = color_groups[0][0]  # Assuming the first color in the group is representative
# #                     else:
# #                         board_color_pattern[i, j] = color_groups[1][0]  # Assuming the first color in the group is representative
# #             break




# middle_indices = [4] if board_size == 8 else [4]

# def is_alternating_pattern(color_pattern, threshold= 50):
#     # Check if the color pattern is alternating between two colors
#     if len(color_pattern) < 2:
#         return False
    
#     # Group similar colors
#     color_groups = group_similar_colors(color_pattern, threshold)
    
#     print(color_groups)
#     # Ensure there are exactly two color groups
#     print(len(color_groups))
#     if len(color_groups) != 2:
#         print("ERRRRRRRRR")
#         return False
    
#     # Check if the colors alternate correctly
#     for i in range(1, len(color_pattern)):
#         if are_colors_similar(color_pattern[i], color_pattern[i-1], threshold):
#             return False
    
#     return True

# # This function is used to predict the entire board based on the found pattern
# def predict_board_color_pattern(board_size, color_groups):
#     # Initialize the board color pattern with zeros
#     board_color_pattern = np.zeros((board_size, board_size, 3), dtype=int)
    
#     print(len(color_groups))
#     # Check if we have exactly two color groups
#     if len(color_groups) == 2:
#         # Assign colors to the board color pattern based on the checkerboard pattern
#         for i in range(board_size):
#             for j in range(board_size):
#                 if (i + j) % 2 == 0:
#                     board_color_pattern[i, j] = color_groups[0][0]  # First color group
#                 else:
#                     board_color_pattern[i, j] = color_groups[1][0]  # Second color group
#     else:
#         # Handle the error case where we do not have two color groups
#         print("Error: Did not find exactly two color groups.")
#         # You can handle this error as appropriate for your application
#         # For example, you might want to return None or raise an exception

#     return board_color_pattern


# # Now, use the functions in the loop to check the middle rows and columns
# for axis in [0, 1]:  # 0 for rows, 1 for columns
#     for idx in middle_indices:
#         color_pattern = []
#         for i in range(board_size):
#             # Calculate the index for the board_color_array
#             contour_idx = idx * board_size + i if axis == 0 else i * board_size + idx
#             major_color = board_color_array[contour_idx // board_size][contour_idx % board_size]
#             print(f"idx : {contour_idx // board_size} idx2 : {contour_idx % board_size}")
#             color_pattern.append(major_color)
        
#         if is_alternating_pattern(color_pattern):
#             color_groups = group_similar_colors(color_pattern)
#             board_color_pattern = predict_board_color_pattern(board_size, color_groups)
#             break  # Found a valid pattern, no need to check further


# # print(board_color_pattern)




# # Now you have an 8x8 or 10x10 array with the predicted color pattern of each square

# # Show the image with detected green line contourf
# show_with_matplotlib(warped_image, 'selected size?')

# # print(board_color_array)
# # Example usage:
# # board_pattern = get_checkerboard_pattern(warped_image, t)
# # print(board_pattern)









# #####this code below board_size must decided.


# # Create an empty image to draw the checkerboard pattern
# square_size = 50  # Size of each square in the checkerboard
# checkerboard_pattern = np.zeros((board_size * square_size, board_size * square_size, 3), dtype=np.uint8)
# checkerboard_image = np.zeros((board_size * square_size, board_size * square_size, 3), dtype=np.uint8)


# # Fill each square with the corresponding color
# for i in range(board_size):
#     for j in range(board_size):
#         top_left = (j * square_size, i * square_size)
#         bottom_right = ((j + 1) * square_size, (i + 1) * square_size)
#         color = tuple(int(c) for c in board_color_pattern[i, j])
#         cv2.rectangle(checkerboard_pattern, top_left, bottom_right, color, -1)

# # Display the checkerboard pattern
# show_with_matplotlib(checkerboard_pattern, 'pattern expected')

# for i in range(board_size):
#     for j in range(board_size):
        
#         top_left = (j * square_size, i * square_size)
#         bottom_right = ((j + 1) * square_size, (i + 1) * square_size)
#         color = tuple(int(c) for c in board_color_array[i, j])
#         cv2.rectangle(checkerboard_image, top_left, bottom_right, color, -1)

# show_with_matplotlib(checkerboard_image, 'original color')


# #need color diff compare code, check them, if there also pixel with circle, then piece.
# print(f'its {board_size} x {board_size}')

# def is_ellipse_almost_circle(ellipse, circle_threshold=0.5):
#     (center, axes, orientation) = ellipse
#     major_axis, minor_axis = max(axes), min(axes)
#     ratio = minor_axis / major_axis
#     return ratio < 1 + circle_threshold and ratio > 1 - circle_threshold

# def is_ellipse_not_too_small(ellipse, min_size_threshold=50):
#     (center, axes, orientation) = ellipse
#     major_axis, minor_axis = axes
#     return major_axis > min_size_threshold and minor_axis > min_size_threshold


# def create_ellipse_mask(shape, ellipse):
#     mask = np.zeros(shape, dtype=np.uint8)
#     cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
#     return mask

# def get_brightness(color):
#     # Assuming color is in BGR format
#     return np.median(color)

# def classify_color(color):
#     brightness = get_brightness(color)
#     return 'w' if brightness > 127 else 'b'  # 127 is the midpoint of 255

# def update_pieces_dict(pieces_dict, color):
#     key = classify_color(color)
#     if key in pieces_dict:
#         pieces_dict[key] += 1
#     else:
#         pieces_dict[key] = 1

# piece_color = []
# pieces = {}
# for i, con in enumerate(t):
#     if (i > board_size * board_size): break
#     x, y, w, h = cv2.boundingRect(con) #if x y is almost mid, get color : save color changes as array : if color != central color and bounding circle
#     # warped_image[x + int(w/2) #color pick, if color is not fit : piece, no : find curve in this area -> sth detected : piece no : nopiece.
#     center_x = x + w//2
#     center_y = y + h//2
#     mask = np.full((1000, 1000),255, dtype = np.uint8)
    
#     cv2.drawContours(mask, [con], -1, 255, -1)
#     # show_with_matplotlib(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 'mask')
#     # res = cv2.copyTo(warped_image, mask)
#     # adjust = 20
#     # y1 = y - adjust if y > adjust else 0
#     # y2 = y + h + adjust if y + h + adjust < 1000 else 1000
#     # x1 = x - adjust if x > adjust else 0
#     # x2 = x + w + adjust if x + h + adjust < 1000 else 1000
#     part2 = warped_image[y:y+h, x:x+w]
#     part = warped_image_gray[y:y+h, x:x+w]
#     maskpiece = mask[y:y+h, x:x+w]
#     # part2 = warped_image[y1:y2, x1:x2]
#     # part = warped_image_gray[y1:y2, x1:x2]
#     # maskpiece = mask[y1:y2, x1:x2]
    
#     # Create a new image to place the extracted contour
#     contour_extracted = np.zeros_like(part)
#     contour_extracted[maskpiece == 255] = part[maskpiece == 255]

#     circles = cv2.HoughCircles(contour_extracted, cv2.HOUGH_GRADIENT, dp = 1.2, minDist = 20, param1=50, param2=30, minRadius=200, maxRadius=1200)
    
#     cont_insquare, _ = cv2.findContours(contour_extracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
#     # show_with_matplotlib(cv2.cvtColor(contour_extracted, cv2.COLOR_GRAY2BGR), 'testf')
    
    
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype('int')
#         for (x, y, r) in circles:
#             cv2.circle(warped_image, (x, y), r, (0, 255, 0), 4)
#     for s in cont_insquare:     #USEFUL    biggest left, also almost circle. color difference and this will figureout piece.
#         if len(s) >= 5:  
#             ellipse = cv2.fitEllipse(s)
#             print(ellipse)
#             a, b, c = ellipse
#             if not is_ellipse_not_too_small(ellipse):
#                 continue
#             # print(type/(b[0]))
#             # if not math.isnan(b[0]) and math.isnan(b[1]):
#                 # Ellipse axes lengths are valid, proceed with drawing
#             cv2.ellipse(part2, ellipse, (0, 255, 0), 2)
#             ellipse_mask = create_ellipse_mask(part2.shape[:2], ellipse)
            
#             # print(f'{i}, {ellipse}')
#             center = ellipse[0]

#             if is_ellipse_almost_circle(ellipse):
#                 # Get the major color inside the ellipse using the ellipse mask
#                 # major_color = get_major_color(part2, ellipse_mask)
#                 mean_color = get_median_color(part2, ellipse_mask)
#                 # print(major_color)
#                 # Determine the position of the contour in the board_color_pattern
#                 row, col = i // board_size, i % board_size
#                 expected_color = board_color_pattern[row, col]
                
#                 update_pieces_dict(pieces, mean_color)
#                 # cv2.ellipse(part2, ellipse, (255, 0, 0), 1)
#                 # if not are_colors_similar(major_color, expected_color):
#                 # if i in piece_color:
#                 #     pieces[tuple(major_color)][0] += 1
#                 #     pieces[tuple(major_color)][1].append((row, col))
#                 # else :
#                 #     piece_color.append(tuple(major_color))
#                 #     pieces[tuple(major_color)] = [1, [(row, col)]]
#                 cv2.ellipse(part2, ellipse, (0, 0, 255), 2)
#                 # show_with_matplotlib(part2, 'hello')
#                 break
    
# # circles = cv2.HoughCircles(warped_image_binary, cv2.HOUGH_GRADIENT, dp = 1.2, minDist = 20, param1=50, param2=30, minRadius=30, maxRadius=50)

# # if circles is not None:
# #     circles = np.round(circles[0, :]).astype('int')
# #     for (x, y, r) in circles:
# #         cv2.circle(warped_image, (x, y), r, (255, , 0), 4)

# # ctrs, _ = cv2.findContours(warped_image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




# # def fit_rotated_ellipse(data):

# # 	xs = data[:,0].reshape(-1,1) 
# # 	ys = data[:,1].reshape(-1,1)

# # 	J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float64))) )
# # 	Y = np.mat(-1*xs**2)
# # 	P= (J.T * J).I * J.T * Y

# # 	a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0]
# #         # To do implementation
# #         #a,b,c,d,e,f 를 통해 theta, 중심(cx,cy) , 장축(major), 단축(minor) 등을 뽑아 낼 수 있어요

# # 	theta = 0.5* np.arctan(b/(a-c))  
# # 	cx = (2*c*d - b*e)/(b**2-4*a*c)
# # 	cy = (2*a*e - b*d)/(b**2-4*a*c)
# # 	cu = a*cx**2 + b*cx*cy + c*cy**2 -f
# # 	w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
# # 	h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

# # 	return (cx,cy,w,h,theta)






# # for con in ctrs:
# #     approx = cv2.approxPolyDP(con, 0.01*cv2.arcLength(con, True), True)
# #     area = cv2.contourArea(con)
# #     if (len(approx) > 10 and area > 400) :
# #         cx, cy, w, h, theta = fit_rotated_ellipse(con.reshape(-1, 2))
# #         cv2.ellipse(warped_image, (int(cx), int(cy)), (int(w), int(h)), theta * 180/np.pi, 0 ,360, (255, 0, 0), 2)

# show_with_matplotlib(warped_image, 'plz')
# for i in pieces.keys():
#     print(f'{i} : {pieces[i]} ')
# # show_with_matplotlib(cv2.cvtColor(warped_image_gray, cv2.COLOR_GRAY2BGR), 'e')





import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title):
    """Helper function to display an image with Matplotlib."""
    img_RGB = color_img[:, :, ::-1]  # Convert BGR to RGB
    plt.figure(figsize=(10, 10))  # Set the figure size to 10x10 inches
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()


# Load images
main_image = cv2.imread('building1.jpg')
template_image = cv2.imread('template.png')  # If you're using a template

# Convert to grayscale
gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)  # If using template

# Feature detection
detector = cv2.SIFT_create()  # Or another feature detector like ORB, SURF
keypoints_main, descriptors_main = detector.detectAndCompute(gray_main, None)
keypoints_template, descriptors_template = detector.detectAndCompute(gray_template, None)  # If using template

# Feature matching (if using template)
matcher = cv2.BFMatcher()  # Or FLANN matcher
matches = matcher.knnMatch(descriptors_template, descriptors_main, k=2)  # If using template

# Filter matches and find homography (if using template)
# ...

# Filter out good matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Find homography if enough good matches are detected
if len(good_matches) > 4:
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches found - %d/%d" % (len(good_matches), 4))
    exit()


# Get the dimensions of the template image
h, w = template_image.shape[:2]

# Project the corners of the template image to the main image
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, matrix)

# Draw the projected corners in the main image
main_image = cv2.polylines(main_image, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)



# Locate and mark the building
# ...

# Display the result

show_with_matplotlib(main_image, 'res')
# cv2.imshow('Detected Building', main_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
