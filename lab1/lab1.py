import cv2 as cv
import numpy as np
import sys

# Constants
MEAN = 0
WEIGHTED_MEAN = 1

# Creates a nxn matrix with custom distribution, given amplitude
def distrib(n, amp):
    kernel = np.empty((n, n), dtype=int)
    factor = amp / 2
    dsum = amp
    for a in kernel:
        for b in a:
            if a == 0 and b == 0:
                kernel[a][b] = amp
            else:
                item = factor / (abs(a) + abs (b))
                kernel[a][b] = item
                dsum = dsum + item
    kernel = kernel / dsum
    return kernel

# Performs 2D convolution using a given mxn kernel
# where m,n MUST be odd numbers
# Function returns a new matrix and does not alter the original
# Args: @matrix @kernel
def convolve(m, k):
    # Check that kernel dimensions are odd numbers
    kx, ky = k.shape
    if kx % 2 != 1 or ky % 2 != 1:
        print("error: convolve: kernel dimensions must be odd numbers;",
                  "got", kx, "and", ky)
        sys.exit(2)

    # Compute what padding is required
    px = kx // 2
    py = ky // 2

    # Pad matrix to ensure successful convolution
    result = pad_matrix(m, 0, px, py, px, py)

    # Loop through the corresponding elements from the input matrix
    # and convolve by replacing the value with the one-to-one multiplication
    # of its neighbouring values and itself with elements from the kernel
    nx, ny = m.shape
    for i in range(0, nx):
        for j in range(0, ny):
            value = 0
            for a in range(0, kx):
                for b in range(0, ky):
                    value = value + result[i + a, j + b] * k[a, b]
            result[i+px, j+py] = value

    # abc
    print(m)

    # Return part of the padded matrix that forms the output
    return result[px:nx+px, py:ny+py]

# Creates padding around a nxn matrix with given value
# Args: @matrix, @value
#       @padding_east, @padding_north, @padding_west, @padding_south
def pad_matrix(m, v, e, n, w, s):
    nx, ny = m.shape
    dx = nx + e + w
    dy = ny + n + s
    result = np.full((dx, dy), v, dtype=m.dtype)
    result[w:nx+w, n:ny+n] = m
    return result


if __name__ == "__main__":
    # Lab info
    print("COMP37212 Lab 1 @ k83954ao (Andrei N. Onea)")
    print("OpenCV version " + cv.__version__)

    # Open image in grayscale
    filepath = "kitty.bmp"
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

    # Check for errors
    if img is None:
        print("error: imread: failed to open" + filepath)
        sys.exit(1)

    mean = convolve(img, np.ones((3,3)) / 9)
    weighted = convolve(img, np.array([[0.5,1,0.5],[1,2,1],[0.5,1,0.5]]) / 8)

    cv.imshow("Coursework 1 - mean filter", mean)
    cv.imshow("Coursework 1 - weighted mean filter", weighted)
    cv.waitKey(0)
    cv.destroyAllWindows()
    sys.exit(0)

    #SHIT GOES BEYOND HERE
    kernel = np.ones((n, n), np.float32) / n*n

