import cv2
import numpy as np
import matplotlib
import black

print(cv2.__version__)
img_filepath = r"C:\SEM6\PCPO\p1\cat.jpg"
img2_path = r"C:\SEM6\PCPO\p1\butterfly.jpg"
img_savefilepath = r"C:\SEM6\PCPO\p1\wynik.jpg"
img = cv2.imread(img_filepath, flags=-1)  # 0 monochromatyczne 1 rgb -1 w/alpha
# cv2.namedWindow("obrazdd", cv2.WINDOW_NORMAL)
cv2.imshow(winname="obraz", mat=img)  # wyświetlenie okna z monochromatyczną wersją obrazu
key = cv2.waitKey(0)  # oczekiwanie na wciśnięcie przycisku przez użytkownika
if key == 27:
    cv2.destroyAllWindows()  # usunięcie okna z obrazem i innych, jeżeli były stworzon
else:  # elif key in [83, 115]:  # S or s
    cv2.imwrite(img_savefilepath, img)
    cv2.destroyAllWindows()
print(f"shape: {img.shape}, size: {img.size}, dtype: {img.dtype}")
print(img[50, 50])
img[100:300, 100:500] = [255, 255, 255]
cv2.imshow(winname="Biała Maska", mat=img)
cv2.waitKey(0)
cv2.destroyAllWindows()

b, g, r = cv2.split(m=img)  # macierze kanałów B G R # print(g)
obr = cv2.merge((g, b, b))
cv2.imshow(winname="kompozycja kanałów", mat=obr)
cv2.waitKey(0)
cv2.destroyAllWindows()

fragment = obr[100:1200, 150:700]
cv2.imshow(winname="fragment", mat=fragment)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.imread(img2_path, flags=-1)
fragment2 = np.zeros((781, 1280, 3), dtype=np.uint8)
fragment2[0:781, 0:1200] = img2[0:781, 0:1200]
fragment2[0:781, 1201:1280] = [255, 0, 255]
fragment2 = fragment2.astype("uint8")
cv2.imshow(winname="img2", mat=fragment2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"shape: {fragment2.shape}, size: {fragment2.size}, {fragment2.dtype}")

new_img = cv2.addWeighted(src1=img, alpha=0.6, src2=fragment2, beta=0.4, gamma=0)
cv2.imshow(winname="new_img", mat=new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

obrazek = np.ones((900, 900, 3), np.uint8) * 255
thickness = 75
cv2.ellipse(
    img=obrazek,
    center=(int((250 + 600) / 2), 500 - int((600 - 250) * np.sqrt(3) / 2)),
    axes=(125, 125),
    angle=0,
    startAngle=120 - 360,
    endAngle=60,
    color=(0, 0, 255),
    thickness=thickness,
    lineType=cv2.LINE_AA,
)
cv2.ellipse(
    img=obrazek,
    center=(250, 500),
    axes=(125, 125),
    angle=0,
    startAngle=0,
    endAngle=300,
    color=(0, 255, 0),
    thickness=thickness,
    lineType=cv2.LINE_AA,
)
cv2.ellipse(
    img=obrazek,
    center=(600, 500),
    axes=(125, 125),
    angle=0,
    startAngle=-60,
    endAngle=240,
    color=(255, 0, 0),
    thickness=thickness,
    lineType=cv2.LINE_AA,
)
cv2.putText(
    img=obrazek,
    text="OpenCV",
    org=(100, 820),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=5.5,
    color=(0, 0, 0),
    thickness=12,
)
cv2.imshow(winname="obrazek", mat=obrazek)
cv2.waitKey(0)
cv2.destroyAllWindows()
