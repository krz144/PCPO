import numpy as np
import cv2
from matplotlib import pyplot as plt

print(cv2.__version__)

# ========= zajecia 15.03 histogramy ======= p3 ======== PCPO_4.pdf
# Zalecane jest wykorzystanie funkcji OpenCV do wyznaczenia histogramu, gdyż jest ona około 40-krotnie szybsza

### Plotting Histograms Using Matplotlib (for one channel) in OpenCV
img_filepath = r"D:\SEMESTR6\PCPO\p2\cat.jpg"
img_filepath = r"C:\SEM6\PCPO\p4\cat.jpg"
img_grey = cv2.imread(img_filepath, 0)
img_color = cv2.imread(img_filepath, 1)
# # cv2.imshow("img_grey", img_grey)
# cv2.imshow("img_color", img_color)
# cv2.waitKey(0)
# histogram = cv2.calcHist(
#     [img_grey],  # Obraz wywoływany w twardych nawiasach [] - format: uint8 lub float32
#     [0],  # Kanał, dla którego wyznaczany jest histogram, 0-blue, 1-green, 2-red
#     None,  # Maska, gdy wyznaczony ma być histogram dla fragmentu
#     [256],  # Wielkość histogramu
#     [0, 256],  # Zakres wartości
# )
# plt.plot(histogram, color="grey")
# plt.xlim([0, 256])
# plt.show()

### Plotting Histograms Using Matplotlib (for all channels) in OpenCV
# for i, col in enumerate(["b", "g", "r"]):
#     histogram = cv2.calcHist([img_color], [i], None, [256], [0, 256])
#     plt.plot(histogram, color=col)
#     plt.xlim([0, 256])
# plt.show()

### Plotting Histograms Using Matplotlib in NumPy
# plt.hist(img_grey.ravel(), 256, [0, 256])
# plt.xlim([0, 256])
# plt.show()  # histogram is basically the same as one calculated using OpenCV. But bins will have 257 elements, because Numpy calculates bins as 0-0.99, 1-1.99, 2-2.99 etc. So final range would be 255-255.99. To represent that, they also add 256 at end of bins. But we don't need that 256. Upto 255 is sufficient.

### Histogram Equalization PCPO_4.pdf (5)
# equ = cv2.equalizeHist(img_grey)
# res = np.hstack((img_grey, equ))  # stacking images side-by-side
# # cv2.imshow("img", res) # show image input vs output
# cv2.imshow("Oryginalne zdjecie", img_grey)  # or show image in two separate windows
# cv2.imshow("Zdjecie po wyrownaniu histogramu", equ)
# cv2.startWindowThread()
# print(cv2.waitKey(0))
# cv2.destroyAllWindows()
# for i in range(2):
#     cv2.waitKey(1)

### Histogram adaptacyjny CLAHE
# claheAlgorytm = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
# adaptacyjneWyrownanieHistogramow = claheAlgorytm.apply(img_grey)
# cv2.imshow("Oryginalne zdjecie", img_grey)  # or show image in two separate windows
# cv2.imshow(
#     "Zdjecie po CLAHE adaptacyjneWyrownanieHistogramow histogramu CLAHE",
#     adaptacyjneWyrownanieHistogramow,
# )
# cv2.imshow("Zdjecie po wyrownaniu histogramu", equ)
# cv2.startWindowThread()
# print(cv2.waitKey(0))
# cv2.destroyAllWindows()
# for i in range(2):
#     cv2.waitKey(1)

### Histogram ręcznie.... minmax scaling(?), normalization
# shape = img_grey.shape
# print(img_grey, shape)
# fmax = img_grey.max()
# fmin = img_grey.min()
# new_matrix = ((img_grey - np.ones(shape) * fmin) / (np.ones(shape) * fmax - np.ones(shape) * fmin) * 255).astype(
#     "uint8"
# )
# print(new_matrix)
# resultimage = np.zeros(shape)
# normalizedimage = cv2.normalize(img_grey, resultimage, 0, 255, cv2.NORM_MINMAX)
# print(normalizedimage)

### image saturation (brighter images)
# scalar = np.ones(img_color.shape, dtype="uint8") * 50
# new_image = cv2.add(img_color, scalar)
# cv2.imshow("original image", img_color)
# cv2.imshow("Add 50 to image", new_image)
# cv2.startWindowThread()
# print(cv2.waitKey(0))
# cv2.destroyAllWindows()

### image saturation (darker images)
# scalar = np.ones(img_color.shape, dtype="uint8") * 50
# new_image = cv2.subtract(img_color, scalar)
# cv2.imshow("original image", img_color)
# cv2.imshow("Subtract 50 from image", new_image)
# cv2.startWindowThread()
# print(cv2.waitKey(0))
# cv2.destroyAllWindows()

### changing the contrast and brightness of the image using cv2.convertScaleAbs() method
alpha = 0.75  # Contrast control - to lower the contrast, use 0 < alpha < 1. And for higher contrast use alpha > 1.
beta = 10  # Brightness control – for example a good range for brightness value is [-127, 127]
adjusted = cv2.convertScaleAbs(img_color, alpha=alpha, beta=beta)
cv2.imshow("original image", img_color)
cv2.imshow("adjusted", adjusted)
cv2.startWindowThread()
print(cv2.waitKey(0))
cv2.destroyAllWindows()

### changing the contrast and brightness of the image using cv2.addWeighted() method
contrast = 2  # Contrast control ( 0 to 127)
brightness = -20  # Brightness control (0-100)
out = cv2.addWeighted(img_color, contrast, img_color, 0, brightness)  # beta 0 to effectively only operate on one image
cv2.imshow("original image", img_color)
cv2.imshow("out", out)
cv2.startWindowThread()
print(cv2.waitKey(0))
cv2.destroyAllWindows()

### Global threshold and binarization
ret, thresh1 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_TOZERO_INV)
titles = ["Original Image", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
images = [img_grey, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray", vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

### Adaptive threshold and binarization
img = cv2.medianBlur(img_grey, 5)
at = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Adaptive Mean Thresholding", at)
print(cv2.waitKey(0))
cv2.destroyAllWindows()

### Adaptive mean thresholding
imgArray = np.asarray(img_grey)
ret1, imgThresholded1 = cv2.threshold(imgArray, 64, 255, cv2.THRESH_BINARY)
ret2, imgThresholded2 = cv2.threshold(imgArray, 127, 255, cv2.THRESH_BINARY)
ret3, imgThresholded3 = cv2.threshold(imgArray, 191, 255, cv2.THRESH_BINARY)
plt.figure(1, figsize=(15, 10))
plt.subplot(131)
plt.title("Próg 64")
plt.axis("off")
plt.imshow(imgThresholded1, cmap="gray")
plt.subplot(132)
plt.title("Próg 127")
plt.axis("off")
plt.imshow(imgThresholded2, cmap="gray")
plt.subplot(133)
plt.title("Próg 191")
plt.axis("off")
plt.imshow(imgThresholded3, cmap="gray")
plt.show()
print(cv2.waitKey(0))
cv2.destroyAllWindows()
