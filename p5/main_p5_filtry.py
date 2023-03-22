import numpy as np
import cv2
from scipy import ndimage


def display(func):  # eksperyment, don't ask...
    def wrapper(*args, **kwargs):
        keys = func.__code__.co_varnames[: func.__code__.co_argcount][::-1]
        sorter = {j: i for i, j in enumerate(keys[::-1])}
        values = func.__defaults__[::-1]
        kwa = {i: j for i, j in zip(keys, values)}
        sorted_default_kwargs = {i: kwa[i] for i in sorted(kwa.keys(), key=sorter.get)}
        img = args[0]
        retv = func(*args, **kwargs)
        domyslny = sorted_default_kwargs["display"]
        podany = kwargs["display"] if "display" in kwargs.keys() else "cokur2"
        if podany != "cokur2" and domyslny != podany:
            if not podany:
                return retv
        elif not domyslny:
            return retv
        cv2.imshow("original image", img)
        cv2.imshow(f"dst: {func.__name__}", retv)
        print(cv2.waitKey(0))
        cv2.destroyAllWindows()
        return retv

    return wrapper


@display
def averaging_filter(img, shape=(5, 5), display=True):
    """Averaging (box) filter"""
    return cv2.filter2D(img, ddepth=-1, kernel=np.ones(shape, np.float32) / (shape[0] * shape[1]))


@display
def box_filter(img, shape=(5, 5), display=True):
    """Box (averaging) filter"""
    return cv2.blur(img, shape)


@display
def gaussian_filter(img, shape=(5, 5), display=True):
    """Gaussian filter"""
    return cv2.GaussianBlur(img, shape, sigmaX=0)


@display
def median_filter(img, ksize=5, display=True):
    """Median filter"""
    return cv2.medianBlur(img, ksize=ksize)


@display
def min_filter(img, size=(5, 5), iterations=1, display=True):
    """Min filter - morphological erosion"""
    shape = cv2.MORPH_RECT
    return cv2.erode(img, kernel=cv2.getStructuringElement(shape=shape, ksize=size), iterations=iterations)


@display
def max_filter(img, size=(5, 5), iterations=1, display=True):
    """Max filter - morphological dilation"""
    shape = cv2.MORPH_RECT
    return cv2.dilate(img, kernel=cv2.getStructuringElement(shape=shape, ksize=size), iterations=iterations)


@display
def roberts_cross(img, display=True):
    """img in greyscale; Roberts operator https://en.wikipedia.org/wiki/Roberts_cross"""
    img = img.astype("float64")
    img /= 255.0
    Gx, Gy = np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])
    conv_1, conv_2 = ndimage.convolve(img, Gx), ndimage.convolve(img, Gy)
    return np.absolute(np.sqrt(np.square(conv_1) + np.square(conv_2)))


def prewitt_filter(img, display=True):
    """Prewitt filter"""
    image_gaussian = cv2.GaussianBlur(img, (3, 3), 0)  # co?
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(image_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(image_gaussian, -1, kernely)
    if display:
        cv2.imshow("original image", img)
        cv2.imshow("Prewitt X", img_prewittx)
        cv2.imshow("Prewitt Y", img_prewitty)
        cv2.imshow("Prewitt", img_prewittx + img_prewitty)
        print(cv2.waitKey(0))
        cv2.destroyAllWindows()
    return img_prewittx + img_prewitty


def sobel_filter(img, ksize=5, display=True):
    """Sobel filter"""
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)  # co?
    img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=ksize)
    img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=ksize)
    img_sobel = img_sobelx + img_sobely
    if display:
        cv2.imshow("original image", img)
        cv2.imshow("Sobel X", img_sobelx)
        cv2.imshow("Sobel Y", img_sobely)
        cv2.imshow("Sobel", img_sobel)
        print(cv2.waitKey(0))
        cv2.destroyAllWindows()
    return img_sobel


@display
def laplace_filter(img, ksize=3, display=True):  # ten laplacian tak ma działać?
    """Laplace filter"""
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)  # co?
    return cv2.Laplacian(img_gaussian, cv2.CV_64F, ksize=ksize)


if __name__ == "__main__":
    # img_filepath = r"D:\SEMESTR6\PCPO\p2\cat.jpg"
    img_filepath = r"C:\SEM6\PCPO\p5\cat.jpg"
    img_grey = cv2.imread(img_filepath, 0)
    img_color = cv2.imread(img_filepath, 1)
    averaging_filter(img_color)
    box_filter(img_color)
    gaussian_filter(img_color)
    median_filter(img_color)
    min_filter(img_color, size=(5, 5))
    max_filter(img_color, size=(5, 5))
    roberts_cross(img_grey)
    prewitt_filter(img_grey)
    sobel_filter(img_grey)
    laplace_filter(img_grey)
