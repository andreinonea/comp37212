#!/usr/bin/env python3

import sys
import cv2 as cv
import numpy as np
import scipy.ndimage as sp
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

IMG_DIR = "../../res"
OUT_DIR = "../../out"

BENCHMARKS = [
    "bernie180.jpg",                    #0
    "bernieBenefitBeautySalon.jpeg",    #1
    "BernieFriends.png",                #2
    "bernieMoreblurred.jpg",            #3
    "bernieNoisy2.png",                 #4
    "berniePixelated2.png",             #5
    "bernieShoolLunch.jpeg",            #6
    "brighterBernie.jpg",               #7
    "darkerBernie.jpg"                  #8
]

# Returns relative path to resources folder
def get_img_path(filename, path=IMG_DIR):
    return str(path + "/" + filename);

# Opens image and checks for error
def get_img(filename):
    filepath = get_img_path(filename)
    img = cv.imread(filepath)

    # Check for errors
    if img is None:
        print("error: imread failed to open " + filepath)
        sys.exit(1)

    return img

# Non-maxima suppresion for input matrix with optional thresholding (default 0)
# Keeps only those values that meet the threshold and also are a local maxima
# in a kx by ky moving window. kx and ky must be odd numbers
# Args: @matrix @kernelX, @kernelY, @threshold
def non_maxima_suppresion(m, kx, ky, threshold=0):

    # Check that kernel dimensions are odd numbers
    if kx % 2 != 1 or ky % 2 != 1:
        print("error: convolve: kernel dimensions must be odd numbers;",
                  "got", kx, "and", ky)
        sys.exit(2)

    # Compute what padding is required
    px = kx // 2
    py = ky // 2

    # Pad matrix to deal with edges and corners
    padded = cv.copyMakeBorder(m, py, py, px, px, borderType=cv.BORDER_REFLECT)

    # Initialize result matrix
    nx, ny = m.shape
    result = np.zeros((nx, ny))

    # Loop through the corresponding elements from the input matrix
    # and check if current value meets the threshold and is a local maxima
    count = 0
    for i in range(0, nx):
        for j in range(0, ny):
            if m[i,j] >= threshold and m[i,j] >= np.max(padded[i:i+kx,j:j+ky]):
                result[i, j] = m[i, j]
                count += 1
    return count, result

# Performs Harris point detector on the specified image
# Image must be grayscale before using it
# Args: @gray: the image, which is assumed to be CV_8U
#                 !!!!WARN! NO CHECK IS DONE!!!!
#       @sigma: value to be used for Gaussian bluring
#       @alpha: value to be used for the factor of trace
#                   when computing corner response
#       @threshold: value to be used when computing NMS
#       @nms: window size on both axes for NMS
def harris_points_detector(gray, sigma=3, alpha=0.04, threshold=0, nms=7):

    # Normalize image to binary64 for increased precision
    gray64 = cv.normalize(gray, None, alpha=0, beta=1,
                           norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)


    # Compute image gradients on X and Y
    # OpenCV Sobel function also blurs the image
    gradient_x64 = cv.Sobel(gray64, cv.CV_64F, 1, 0, ksize=3,
                            borderType=cv.BORDER_REFLECT)
    gradient_y64 = cv.Sobel(gray64, cv.CV_64F, 0, 1, ksize=3,
                            borderType=cv.BORDER_REFLECT)

    # Compute the angles of the gradient at each point
    orientations = np.degrees(np.arctan2(gradient_y64, gradient_x64))

    # Compute the combinations between the gradients
    # i.e Ix^2, Iy^2, Ixy
    grad_xx64 = np.square(gradient_x64)
    grad_yy64 = np.square(gradient_y64)
    grad_xy64 = np.multiply(gradient_x64, gradient_y64)

    # Blur those
    grad_xx_blur64 = sp.gaussian_filter(grad_xx64, sigma)
    grad_yy_blur64 = sp.gaussian_filter(grad_yy64, sigma)
    grad_xy_blur64 = sp.gaussian_filter(grad_xy64, sigma)

    # 2x2 M matrix determinat and trace calculation
    det = np.subtract(np.multiply(grad_xx_blur64, grad_yy_blur64),
                      np.square(grad_xy_blur64))
    trace = np.add(grad_xx_blur64, grad_yy_blur64)

    # Compute corner response and normalize it back to uint8
    R64 = np.subtract(det, np.multiply(alpha, np.square(trace)))
    R = cv.normalize(R64, None, alpha=0, beta=256,
                     norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Perform non-maxima suppresion and thresholding on R
    _, R_nms = non_maxima_suppresion(R, nms, nms, threshold=threshold)

    # For viewing purposes, make a version with positive values all white
    _, R_nms_white = cv.threshold(R_nms, 0, 255, cv.THRESH_BINARY)

    # Create keypoints from filtered corner response R
    keypoints = []
    rx, ry = R_nms.shape
    for i in range(0, rx):
        for j in range(0, ry):
            if R_nms[i, j] > 0:
                keypoints.append(cv.KeyPoint(j, i, nms, orientations[i, j]))

    gx = cv.normalize(R64, None, alpha=0, beta=256,
                           norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #plt.imshow(R, cmap='hot')
    #plt.show()

    return R, keypoints

# SSD - sum of squared differences between two descriptors
# also performs ratio test; if ratio is closed to 1, the feature is
# ambiguous so it is discarded
def SSD(a, b):
    matches = []

    dists = cdist(a, b)
    nx, ny = dists.shape
    for i in range(0, nx):
        indices = np.argsort(dists[i])
        idx = indices[0]
        idx2 = indices[1]
        first = dists[i, idx]
        second = dists[i, idx2]
        ratio = first / second
        if first < SSD_THRESHOLD and ratio < SSD_RATIO_THRESHOLD:
            matches.append(cv.DMatch(i, idx, first))

    return matches


# Program entry point
if __name__ == "__main__":
    # Lab info
    print("COMP37212 Lab 2 @ k83954ao (Andrei N. Onea)")
    print("OpenCV version " + cv.__version__)

    # Open reference image
    img = get_img("bernieSanders.jpg")

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Init ORB object
    orb = cv.ORB_create()

    # Non-maxima suppresion window size
    NMS = 7
    # Threshold values
    NMS_THRESHOLD=60
    SSD_THRESHOLD=200
    SSD_RATIO_THRESHOLD=0.9

    # Find interest points using OpenCV ORB corner detector
    kp = orb.detect(gray, None)
    # Compute descriptors for ORB keypoints
    kp, _ = orb.compute(gray, kp)
    # Draw keypoints
    gray_kp = cv.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)
    #cv.imshow("Orb", gray_kp)

    # Find interest points using own Harris corner detector
    R, keypoints = harris_points_detector(gray, nms=NMS,
                                          threshold=NMS_THRESHOLD)

    # Compute descriptors for own keypoints
    keypoints, descriptors = orb.compute(gray, keypoints)

    # Draw keypoints
    gray_keypoints = cv.drawKeypoints(gray, keypoints,
                                      None, color=(0, 255, 0), flags=0)
    cv.imwrite(get_img_path("orig_keypoints.jpg", path=OUT_DIR),
               gray_keypoints)
    comparison = np.concatenate((gray_keypoints, gray_kp), axis=1)
    cv.imwrite(get_img_path("comparison.jpg", path=OUT_DIR), comparison)

    # Try to find matches for every benchmark reference image
    for filename in BENCHMARKS:
        bench = get_img(filename)
        bench_gray = cv.cvtColor(bench, cv.COLOR_BGR2GRAY)
        bench_R, bench_kp = harris_points_detector(bench_gray, nms=NMS,
                                                   threshold=NMS_THRESHOLD)
        bench_kp, bench_des = orb.compute(bench_gray, bench_kp)
        bench_graykp = cv.drawKeypoints(bench_gray, bench_kp,
                                        None, color=(0,255,0), flags=0)

        matches = SSD(descriptors, bench_des)
        gray_matches = cv.drawMatches(gray, keypoints,
                                      bench_gray, bench_kp,
                                      matches,
                                      None)
        out = get_img_path(filename + "_keypoints.jpg", path=OUT_DIR)
        cv.imwrite(out, bench_graykp)
        out = get_img_path(filename + "_match.jpg", path=OUT_DIR)
        cv.imwrite(out, gray_matches)

    # Get data for plotting keypoints count against threshold value
    #for i in range(45, 256):
    #    count, _ = non_maxima_suppresion(R, NMS, NMS, i)
    #    print(i, " ", count)

    #cv.imshow("Match", gray_matches)

    #cv.waitKey(0)
    #cv.destroyAllWindows()
