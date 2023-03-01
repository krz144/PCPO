import numpy as np
import matplotlib
import cv2

print(cv2.__version__)

img_filepath = r"D:\SEMESTR6\PCPO\p1\cat.jpg"
img_savefilepath = r"D:\SEMESTR6\PCPO\p1\wynik.jpg"
img = cv2.imread(img_filepath, flags=-1)  # 0 monochromatyczne 1 rgb -1 w/alpha
# cv2.namedWindow("obrazdd", cv2.WINDOW_NORMAL)
cv2.imshow(
    winname="obraz", mat=img
)  # wyświetlenie okna z monochromatyczną wersją obrazu
key = cv2.waitKey(0)  # oczekiwanie na wciśnięcie przycisku przez użytkownika
if key == 27:
    cv2.destroyAllWindows()  # usunięcie okna z obrazem i innych, jeżeli były stworzon
else:  # elif key in [83, 115]:  # S or s
    cv2.imwrite(img_savefilepath, img)
    cv2.destroyAllWindows()


print(f"shape: {img.shape}, size: {img.size}")
print(img[50, 50])
# print(img)


img[100:300, 100:500] = [255, 255, 255]
cv2.imshow(winname="Biała Maska", mat=img)
cv2.waitKey(0)
cv2.destroyAllWindows()

b, g, r = cv2.split(m=img)  # macierze kanałów B G R
# print(g)
obr = cv2.merge((g, b, b))
cv2.imshow(winname="kompozycja kanałów", mat=obr)
cv2.waitKey(0)
cv2.destroyAllWindows()

fragment = obr[100:1200, 150:700]
cv2.imshow(winname="fragment", mat=fragment)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2_path = r"D:\SEMESTR6\PCPO\p1\butterfly.jpg"
img2 = cv2.imread(img2_path, flags=-1)
fragment2 = img2[0:781, 0:1280]
cv2.imshow(winname="img2", mat=fragment2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"shape: {fragment2.shape}, size: {fragment2.size}")
# print(fragment2)

# WTF nie aktualilzuje wycinka
# new_img = cv2.addWeighted(src1=img, alpha=0.6, src2=fragment2, beta=0.4, gamma=20)
# cv2.imshow(winname="new_img", mat=new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


obrazek = np.ones((800, 1000, 3), np.uint8) * 255
cv2.ellipse(
    img=obrazek,
    center=(500, 250),
    axes=(125, 125),
    angle=0,
    startAngle=-210,
    endAngle=90,
    color=(0, 0, 255),
    thickness=100,
)
cv2.ellipse(
    img=obrazek,
    center=(250, 600),
    axes=(125, 125),
    angle=0,
    startAngle=0,
    endAngle=310,
    color=(0, 255, 0),
    thickness=100,
)
cv2.ellipse(
    img=obrazek,
    center=(750, 600),
    axes=(125, 125),
    angle=0,
    startAngle=0,
    endAngle=310,
    color=(255, 0, 0),
    thickness=100,
)
cv2.imshow(winname="obrazek", mat=obrazek)
cv2.waitKey(0)
cv2.destroyAllWindows()
