import numpy as np
import cv2
import math

print(cv2.__version__)

# img_filepath = r"D:\SEMESTR6\PCPO\p2\cat.jpg"
img_filepath = r"C:\SEM6\PCPO\p2\cat.jpg"

# ============== rysowanie prostokątu obsługa myszki ===========
# points = []
# img = cv2.imread(img_filepath, 1)
# cv2.namedWindow("image")
# # mouse callback function
# def draw_rectangle(event, x, y, flags, param):
#     global points
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         points = [(x, y)]
#     elif event == cv2.EVENT_RBUTTONDBLCLK:
#         points.append((x, y))
#         if np.size(points, 0) == 2:
#             cv2.rectangle(img, points[0], points[1], (200, 200, 0), 2)
#         else:
#             print("Not enough number of points")


# cv2.setMouseCallback("image", draw_rectangle)
# while True:
#     cv2.imshow("image", img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# ============================================= hold mouse rect ===========================
img = cv2.imread(img_filepath, 1)
img2 = img.copy()
drawing = False  # True if mouse is pressed
s_x, s_y = -1, -1
# mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global img, s_x, s_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        s_x, s_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = img2.copy()
            cv2.rectangle(img, (s_x, s_y), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (s_x, s_y), (x, y), (0, 255, 0), 2)


cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_rectangle)
while True:
    cv2.imshow("image", img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
# ============================================= hold mouse rect ===========================

# ========================= ODBICIA GÓRA DÓŁ, LEWA PRAWA ============
# zdjęcie = wczytane zdjęcie
# zdjecie = img

# # Przygotowanie mapy  przekształcającej zdjęcie
# mapaPrzeksztalcenX = np.zeros((zdjecie.shape[0], zdjecie.shape[1]), dtype=np.float32)
# mapaPrzeksztalcenY = np.zeros((zdjecie.shape[0], zdjecie.shape[1]), dtype=np.float32)
# for i in range(mapaPrzeksztalcenX.shape[0]):
#     mapaPrzeksztalcenX[i, :] = [x for x in range(mapaPrzeksztalcenX.shape[1])]
# for j in range(mapaPrzeksztalcenY.shape[1]):
#     mapaPrzeksztalcenY[:, j] = [
#         mapaPrzeksztalcenY.shape[0] - y for y in range(mapaPrzeksztalcenY.shape[0])
#     ]

# # Przekształcenie obrazu
# wynikowyObraz = cv2.remap(
#     zdjecie, mapaPrzeksztalcenX, mapaPrzeksztalcenY, cv2.INTER_LINEAR
# )

# # Wyświetalnie obrazu
# cv2.imshow("Oryginalne - góra", zdjecie)
# cv2.imshow("Nowe zdjęcie - dół", wynikowyObraz)
# cv2.waitKey(0)

# # lewa-prawa
# # Przygotowanie mapy  przekształcającej zdjęcie
# mapaPrzeksztalcenX = np.zeros((zdjecie.shape[0], zdjecie.shape[1]), dtype=np.float32)
# mapaPrzeksztalcenY = np.zeros((zdjecie.shape[0], zdjecie.shape[1]), dtype=np.float32)
# for i in range(mapaPrzeksztalcenX.shape[0]):
#     mapaPrzeksztalcenX[i, :] = [
#         mapaPrzeksztalcenY.shape[1] - x for x in range(mapaPrzeksztalcenX.shape[1])
#     ]
# for j in range(mapaPrzeksztalcenY.shape[1]):
#     mapaPrzeksztalcenY[:, j] = [y for y in range(mapaPrzeksztalcenY.shape[0])]

# # Przekształcenie obrazu
# wynikowyObraz = cv2.remap(
#     zdjecie, mapaPrzeksztalcenX, mapaPrzeksztalcenY, cv2.INTER_LINEAR
# )

# # Wyświetalnie obrazu
# cv2.imshow("Oryginalne - lewa", zdjecie)
# cv2.imshow("Nowy obraz - prawa", wynikowyObraz)
# cv2.waitKey(0)


# ========================= zniekształcenie dystorsja radialna  ============
# zdjecie = cv2.imread(img_filepath, 1)
# znieksztalcenie = 3
# # zapis do zmiennej wymiarów obrazu
# (wysokosc, szerokosc, _) = zdjecie.shape
# # Zdefiniowanie map przekształceń współrzędnych w formaciefloat32
# mapaPrzeksztalcenX = np.zeros((wysokosc, szerokosc), np.float32)
# mapaPrzeksztalcenY = np.zeros((wysokosc, szerokosc), np.float32)
# wspolrzędnaXsrodka = szerokosc / 2
# wspolrzędnaYsrodka = wysokosc / 2
# promienNormalizujacy = szerokosc / 2

# for y in range(wysokosc):
#     deltaY = y - wspolrzędnaYsrodka
#     for x in range(szerokosc):
#         deltaX = x - wspolrzędnaXsrodka
#         odlegloscOdSrodka = np.power(deltaX, 2) + np.power(deltaY, 2)
#         if odlegloscOdSrodka >= np.power(promienNormalizujacy, 2):
#             mapaPrzeksztalcenX[y, x] = x
#             mapaPrzeksztalcenY[y, x] = y
#         else:
#             wspolczynnikiZnieksztalcenia = 1.0  # znieksztalcenie = 1 ??
#         if odlegloscOdSrodka > 0.0:
#             wspolczynnikiZnieksztalcenia = math.pow(
#                 math.sin(
#                     math.pi * math.sqrt(odlegloscOdSrodka) / promienNormalizujacy / 2
#                 ),
#                 znieksztalcenie,
#             )
#         mapaPrzeksztalcenX[y, x] = (
#             wspolczynnikiZnieksztalcenia * deltaX + wspolrzędnaXsrodka
#         )
#         mapaPrzeksztalcenY[y, x] = (
#             wspolczynnikiZnieksztalcenia * deltaY + wspolrzędnaYsrodka
#         )

# dst = cv2.remap(zdjecie, mapaPrzeksztalcenX, mapaPrzeksztalcenY, cv2.INTER_LINEAR)

# cv2.imshow("Zniekształcone", dst)
# cv2.waitKey(0)


# ================= translacja z użyciem warpAffine() =============
# zdjecie = cv2.imread(img_filepath, 1)
# wysokosc, szerokosc, _ = zdjecie.shape
# macierzTranslacji = np.float32([[1, 0, 50], [0, 1, 150]])
# print(macierzTranslacji)
# zdjeciePoTranslacji = cv2.warpAffine(zdjecie, macierzTranslacji, (wysokosc, szerokosc))
# cv2.imshow("Oryginalne", zdjecie)
# cv2.imshow("Nowy obraz", zdjeciePoTranslacji)
# cv2.waitKey(0)
# (wysokosc, szerokosc, _) = zdjecie.shape
# macierzTranslacji = np.float32([[1, 0, 50], [0, 1, 150], [0, 0, 1]])
# print(macierzTranslacji)
# zdjeciePoTranslacji = cv2.warpPerspective(zdjecie, macierzTranslacji, (wysokosc, szerokosc))
# cv2.imshow("Oryginalne", zdjecie)
# cv2.imshow("Nowy obraz", zdjeciePoTranslacji)
# cv2.waitKey(0)

# ===================== skalowanie obrazu =========================
zdjecie = cv2.imread(img_filepath, 1)
wysokosc, szerokosc, _ = zdjecie.shape
zdjeciePoSkalowaniu = cv2.resize(zdjecie, (int(szerokosc / 2), int(wysokosc / 2)))
cv2.imshow("zdjeciePoSkalowaniu", zdjeciePoSkalowaniu)
cv2.waitKey(0)

zdjeciePoSkalowaniu = cv2.resize(zdjecie, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow("zdjeciePoSkalowaniu", zdjeciePoSkalowaniu)
cv2.waitKey(0)


# ===================== transformacja shear ========================
zdjecie = cv2.imread(img_filepath, 1)
wysokosc, szerokosc, _ = zdjecie.shape
print(wysokosc, szerokosc)
macierzTransformacji = np.float32([[1, -0.5, 0], [-0.4, 1, 0], [0, 0, 1]])
LD = macierzTransformacji @ np.float32([0, 0, 1])
RU = macierzTransformacji @ np.float32([szerokosc, wysokosc, 1])
LU = macierzTransformacji @ np.float32([0, wysokosc, 1])
RD = macierzTransformacji @ np.float32([szerokosc, 0, 1])
wysokosc = abs(min([c[0] for c in [LD, RU, LU, RD]])) + abs(max([c[0] for c in [LD, RU, LU, RD]]))
szerokosc = abs(min([c[1] for c in [LD, RU, LU, RD]])) + abs(max([c[1] for c in [LD, RU, LU, RD]]))
dx = int(-1 * min([c[0] for c in [LD, RU, LU, RD]]))
dy = int(-1 * min([c[1] for c in [LD, RU, LU, RD]]))
macierzTransformacji[0][2] = dx
macierzTransformacji[1][2] = dy
zdjeciePoPochyleniu = cv2.warpPerspective(zdjecie, macierzTransformacji, (int(wysokosc + 0.5), int(szerokosc + 0.5)))
cv2.imshow("Shear mapping", zdjeciePoPochyleniu)
cv2.waitKey(0)
