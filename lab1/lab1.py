import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

# Switch-like function to return an OpenCV type from a Numpy type
# Thanks to: https://stackoverflow.com/questions/60208/replacements-for-switch-statement-in-python
def np_to_cv_type(x):
    return {
        np.uint8: (cv.CV_8U, 0, 255),
        np.float32: (cv.CV_32F, 0, 1),
        np.float64: (cv.CV_64F, 0, 1)
    }.get(x, (cv.CV_8U, 0, 255))

# Generates an average distribution mxn matrix
# in float32 precision by default
# Args: @rows, @columns, @type
def mean_distribution2D(row, col, ktype=np.float32):
    return np.ones((row, col), dtype=ktype) / (row * col)

# Generates a gaussian distribution mxn matrix
# in float32 by default, with the center at (0,0)
# Normalization should usually take place
# Args: @rows, columns, @amplitude,
#       @x spread, @y spread, @x center, @y center,
#       @type, @normalization
def gaussian_distribution2D(row, col, amp, sx, sy,
                            cx=0, cy=0,
                            ktype=np.float32,
                            normalize=True):
    # Initialize empty matrix of mxn
    # and use float64 internally
    kernel = np.empty((row, col), dtype=np.float64)

    # Compute anchor point
    ax = row // 2
    ay = col // 2

    # Fill matrix with results
    # from the 2D Gaussian function
    total = 0.0
    for i in range(0, row):
        for j in range(0, col):
            x = i - ax; y = j - ay
            gx = (((x-cx)**2) / 2 * sx**2)
            gy = (((y-cy)**2) / 2 * sy**2)
            value = amp * np.exp(-(gx + gy))
            kernel[i, j] = value
            total = total + value
    # Normalize numbers by default
    # so they add up to 1
    if normalize:
        kernel = kernel / total

    # Finally, return as the specified type
    return kernel.astype(ktype)

# Creates padding around a mxn matrix with given value
# Parameters go based on trigonometrical circle
# Args: @matrix, @value
#       @padding_east, @padding_north,
#       @padding_west, @padding_south
def pad_matrix2D(m, v, e, n, w, s):
    nx, ny = m.shape
    dx = nx + e + w
    dy = ny + n + s
    result = np.full((dx, dy), v, dtype=m.dtype)
    result[w:nx+w, n:ny+n] = m
    return result

# Performs binary thresholding on an image
# Args: @input image, @threshold
def threshold2D_binary(img, threshold):
    nx, ny = img.shape
    result = np.zeros(img.shape, dtype=img.dtype)
    for i in range(nx):
        for j in range(ny):
            if img[i, j] < threshold:
                result[i, j] = 0
            else:
                result[i, j] = 255 if img.dtype == np.uint8 else 1
    return result

# Performs 2D convolution using a given mxn kernel
# where m,n MUST be odd numbers
# Function returns a new matrix
# and does not alter the original
# Internally it uses float64 for best precision
# but preserves the original data type in the end
# Args: @matrix @kernel
def convolve2D(m, k):
    # Check that kernel dimensions are odd numbers
    kx, ky = k.shape
    if kx % 2 != 1 or ky % 2 != 1:
        print("error: convolve: kernel dimensions must be odd numbers;",
                  "got", kx, "and", ky)
        sys.exit(2)

    # Remember the original type
    orig_type, alpha, beta = np_to_cv_type(m.dtype)

    # Compute what padding is required
    px = kx // 2
    py = ky // 2

    # Pad matrix to deal with edges and corners and convert it to float64
    padded = pad_matrix2D(m, 0, px, py, px, py)
    padded64 = cv.normalize(padded, None, alpha=0, beta=1,
                            norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    
    # Initialize result matrix
    result64 = np.empty(m.shape, dtype=np.float64)

    # Loop through the corresponding elements from the input matrix
    # and convolve by replacing the value with the one-to-one multiplication
    # of its neighbouring values and itself with elements from the kernel
    nx, ny = m.shape
    for i in range(0, nx):
        for j in range(0, ny):
            value = 0.0
            for a in range(0, kx):
                for b in range(0, ky):
                    value = value + padded64[i + a, j + b] * k[a, b]
            result64[i, j] = value

    # Return part of the padded matrix that forms the output
    #result = cv.normalize(result64, None, alpha=0, beta=1,
    #                        norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    # Return part of the padded matrix that forms the output
    result = cv.normalize(np.absolute(result64), None, alpha=alpha, beta=beta,
                            norm_type=cv.NORM_MINMAX, dtype=orig_type)
    return result

# Performs mean filtering on the input image
# Args: @input image, @number of rows in kernel, @number of columns in kernel
#       @type
def mean_filter2D(img, krow, kcol, ktype=np.float32):
    kernel = mean_distribution2D(krow, kcol, ktype=ktype)
    return convolve2D(img, kernel)

# Performs Gaussian filtering on the input image
# Args: @input image, @number of rows in kernel, @number of columns in kernel
#       @amplitude, @x spread, @y spread, @x center, @y center
#       @type, @normalization 
def gaussian_filter2D(img, krow, kcol, amp, sx, sy, cx=0, cy=0,
                        ktype=np.float32, normalize=True):
    kernel = gaussian_distribution2D(krow, kcol, amp, sx, sy, cx=cx, cy=cy,
                                        ktype=ktype, normalize=normalize)
    return convolve2D(img, kernel)


# Program entry point
if __name__ == "__main__":
    # Lab info
    print("COMP37212 Lab 1 @ k83954ao (Andrei N. Onea)")
    print("OpenCV version " + cv.__version__)

    # Open image in grayscale
    filepath = "kitty.bmp"
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

    # Check for errors
    if img is None:
        print("error: imread: failed to open", filepath)
        sys.exit(1)

    # Define the Sobel kernels
    sobelX_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    sobelY_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)

    # Define experiment parameters
    GK_SIZE = 7 # Gaussian kernel size
    GK_AMP = 5 # Gaussian kernel amplitude
    GK_SX = 0.25 # Gaussian kernel spread on X axis
    GK_SY = 0.25 # Gaussian kernel spread on Y axis
    THRESH_VALUE = 24
    WINDOW_NAME = 'Lab 1'

    # Start processing

    # Perform experiment for an average filter starting kernel
    # Filter the image, then find the image gradients on X and Y
    # and finally compute the gradient magnitude by combining the two
    # image gradients. A histogram is then generated to find a feasible
    # value for the thresholding, which is the final step
    img_mean_blur = mean_filter2D(img, 7, 7, ktype=np.float64)
    img_mean_sobelX = convolve2D(img_mean_blur, sobelX_kernel)
    img_mean_sobelY = convolve2D(img_mean_blur, sobelY_kernel)
    img_mean_gradient = cv.addWeighted(img_mean_sobelX, 0.5,
                                       img_mean_sobelY, 0.5, 0)

    # Compute histogram
    img_mean_hist = cv.calcHist([img_mean_gradient], [0], None, [256], [0, 256])
    img_mean_hist = img_mean_hist.reshape(256)

    # Threshold to find edges
    img_mean_edges = threshold2D_binary(img_mean_gradient, THRESH_VALUE)

    # Perform experiment for an weighted-mean filter kernel
    # chosen using Gaussian distribution
    img_gaussian_blur = gaussian_filter2D(img, GK_SIZE, GK_SIZE,
        GK_AMP, GK_SX, GK_SY, ktype=np.float64)
    img_gaussian_sobelX = convolve2D(img_gaussian_blur, sobelX_kernel)
    img_gaussian_sobelY = convolve2D(img_gaussian_blur, sobelY_kernel)
    img_gaussian_gradient = cv.addWeighted(img_gaussian_sobelX, 0.5,
                                           img_gaussian_sobelY, 0.5, 0)

    # Compute histogram
    img_gaussian_hist = cv.calcHist([img_gaussian_gradient],[0],None,[256],[0, 256])
    img_gaussian_hist = img_gaussian_hist.reshape(256)

    # Threshold to find edges
    img_gaussian_edges = threshold2D_binary(img_gaussian_gradient, THRESH_VALUE)

    # Compare edge strength images
    img_edge_comparison = img_mean_edges - img_gaussian_edges

    # OpenCV window management
    # Show all images in a single window for easy control with a slider
    horizontal = np.concatenate((img_mean_blur, img_mean_sobelX), axis=1)
    horizontal = np.concatenate((horizontal, img_mean_sobelY), axis=1)
    horizontal = np.concatenate((horizontal, img_mean_gradient), axis=1)
    horizontal = np.concatenate((horizontal, img_mean_edges), axis=1)
    horizontal = np.concatenate((horizontal, img), axis=1)
    vertical = np.concatenate((img_gaussian_blur, img_gaussian_sobelX), axis=1)
    vertical = np.concatenate((vertical, img_gaussian_sobelY), axis=1)
    vertical = np.concatenate((vertical, img_gaussian_gradient), axis=1)
    vertical = np.concatenate((vertical, img_gaussian_edges), axis=1)
    vertical = np.concatenate((vertical, img_edge_comparison), axis=1)
    window = np.concatenate((horizontal, vertical), axis=0)

    # Create a window to display the images
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    # Display results
    cv.imshow(WINDOW_NAME, window)

    # Display histograms
    mean_plot = plt.figure('Mean kernel image histogram')
    plt.bar(np.linspace(0, 255, 256), img_mean_hist)
    plt.title('Histogram')
    plt.title('Gray level')
    plt.ylabel('Frequency')

    gaussian_plot = plt.figure('Weighted-mean kernel image histogram')
    plt.bar(np.linspace(0, 255, 256), img_gaussian_hist)
    plt.title('Histogram')
    plt.title('Gray level')
    plt.ylabel('Frequency')

    plt.show()

    # Wait for key press from user to keep windows alive
    # If user presses 'S' key, the images are saved on disk
    k = cv.waitKey(0)
    if k == ord('s'):
        print('Saving img_mean_blur.jpg')
        cv.imwrite('img_mean_blur.jpg', img_mean_blur)
        print('Saving img_mean_sobelX.jpg')
        cv.imwrite('img_mean_sobelX.jpg', img_mean_sobelX)
        print('Saving img_mean_sobelY.jpg')
        cv.imwrite('img_mean_sobelY.jpg', img_mean_sobelY)
        print('Saving img_mean_gradient.jpg')
        cv.imwrite('img_mean_gradient.jpg', img_mean_gradient)
        print('Saving img_mean_hist.jpg')
        mean_plot.savefig('img_mean_hist.jpg')
        print('Saving img_mean_edges.jpg')
        cv.imwrite('img_mean_edges.jpg', img_mean_edges)
        print('Saving img_gaussian_blur.jpg')
        cv.imwrite('img_gaussian_blur.jpg', img_gaussian_blur)
        print('Saving img_gaussian_sobelX.jpg')
        cv.imwrite('img_gaussian_sobelX.jpg', img_gaussian_sobelX)
        print('Saving img_gaussian_sobelY.jpg')
        cv.imwrite('img_gaussian_sobelY.jpg', img_gaussian_sobelY)
        print('Saving img_gaussian_gradient.jpg')
        cv.imwrite('img_gaussian_gradient.jpg', img_gaussian_gradient)
        print('Saving img_gaussian_hist.jpg')
        mean_plot.savefig('img_gaussian_hist.jpg')
        print('Saving img_gaussian_edges.jpg')
        cv.imwrite('img_gaussian_edges.jpg', img_gaussian_edges)
        print('Saving img_edge_comparison.jpg')
        cv.imwrite('img_edge_comparison.jpg', img_edge_comparison)
    else:
        cv.destroyAllWindows()
    sys.exit(0)
