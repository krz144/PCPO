# openCV mouse events and image transformations
# https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
import cv2
import numpy as np
import math
import imutils


def mouse_callback_function1(event, x, y, flags, param):
    global points, img, img2, img3, drawing, s_x, s_y
    (wysokosc, szerokosc, _) = img.shape

    # Odbicie względem osi X: CTRL + LPM
    if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
        macierzTransformacjiX = np.float32([[1, 0, 0], [0, -1, wysokosc], [0, 0, 1]])
        img = cv2.warpPerspective(img, macierzTransformacjiX, (szerokosc, wysokosc))

    # Odbicie względem osi Y: CTRL + PPM
    elif flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_RBUTTON:
        macierzTransformacjiY = np.float32([[-1, 0, szerokosc], [0, 1, 0], [0, 0, 1]])
        img = cv2.warpPerspective(img, macierzTransformacjiY, (szerokosc, wysokosc))

    # Skalowanie do wybranej przekątnej: ALT + LPM, ALT + PPM
    elif flags == cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_LBUTTON:
        points = [(x, y)]
    elif flags == cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_RBUTTON:
        points.append((x, y))
        width = max([k[0] for k in points]) - min([k[0] for k in points])
        height = max([k[1] for k in points]) - min([k[1] for k in points])
        if np.size(points, 0) == 2:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # Skalowanie względem odcinka
    elif flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON:
        points = [(x, y)]
        drawing = True
        s_x, s_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = img2.copy()
            cv2.rectangle(img, (s_x, s_y), (x, y), (0, 255, 0), 2)
    elif flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_RBUTTON:
        drawing = False
        cv2.rectangle(img, (s_x, s_y), (x, y), (0, 255, 0), 2)
        w = max(x, s_x) - min(x, s_x)
        h = max(y, s_y) - min(y, s_y)
        print(f"w: {w}, h: {h}")
        print("Podaj długość odcinka x (|| oś X) [px] (bez zmian -> wpisz -1): ")
        odc_x = int(input())
        if odc_x >= 0:
            skala_fx = odc_x / w
        else:
            skala_fx = 1
        print(f"skala_fx: {skala_fx}")
        print("Podaj długość odcinka y (|| oś Y) [px] (bez zmian -> wpisz -1): ")
        odc_y = int(input())
        if odc_y >= 0:
            skala_fy = odc_y / h
        else:
            skala_fy = 1
        print(f"skala_fy: {skala_fy}")
        img = img3.copy()
        img = cv2.resize(img, dsize=None, fx=skala_fx, fy=skala_fy, interpolation=cv2.INTER_CUBIC)

    # Obrót
    elif flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON:
        points = [(x, y)]
    elif flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_RBUTTON:
        points.append((x, y))
        if np.size(points, 0) == 2:
            p1, p2 = points[0], points[1]
            katWStopniach = np.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            img = imutils.rotate_bound(img, katWStopniach)

    # Transformacja afiniczna (najpierw wybieramy punkt LPM, potem dodajemy kolejne PPM)
    elif flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_LBUTTON:
        points = [(x, y)]
        print(points)
    elif flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_RBUTTON:
        points.append((x, y))
        print(points)
        if np.size(points, 0) == 3:
            print("podaj 3 punkty referencyjne(?) (x1,y1,x2,y2,x3,y3):")
            p1x = int(input())
            p1y = int(input())
            p2x = int(input())
            p2y = int(input())
            p3x = int(input())
            p3y = int(input())
            print(p1x, p1y, p2x, p2y, p3x, p3y)
            punkty1 = np.float32([points[0], points[1], points[2]])
            punkty2 = np.float32([[p1x, p1y], [p2x, p2y], [p3x, p3y]])
            macierzTransformacji = cv2.getAffineTransform(punkty1, punkty2)
            img = cv2.warpAffine(img, macierzTransformacji, (szerokosc, wysokosc))
            # ^ szerokosc, wysokosc trzeba obliczyc by pokazalo caly obraz...

    # Transformacja perspektywiczna (najpierw wybieramy punkt LPM, potem dodajemy kolejne PPM)
    elif flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_LBUTTON:
        points = [(x, y)]
        print(points)
    elif flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_RBUTTON:
        points.append((x, y))
        print(points)
        if np.size(points, 0) == 4:
            print("podaj 4 punkty referencyjne(?) (x1,y1,x2,y2,x3,y3,x4,y4):")
            p1x = int(input())
            p1y = int(input())
            p2x = int(input())
            p2y = int(input())
            p3x = int(input())
            p3y = int(input())
            p4x = int(input())
            p4y = int(input())
            print(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)
            punkty1 = np.float32([points[0], points[1], points[2], points[3]])
            punkty2 = np.float32([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]])
            macierzTransformacji = cv2.getPerspectiveTransform(punkty1, punkty2)
            img = cv2.warpPerspective(img, macierzTransformacji, (szerokosc, wysokosc))
            # ^ szerokosc, wysokosc trzeba obliczyc by pokazalo caly obraz...
    cv2.imshow("image", img)


if __name__ == "__main__":
    img_path = r"C:\SEM6\PCPO\p2\cat.jpg"
    img = cv2.imread(img_path, 1)
    img2 = img.copy()
    img3 = img.copy()
    drawing = False
    s_x, s_y = -1, -1
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback_function1)
    while True:
        cv2.imshow("image", img)
        if cv2.waitKey(0):
            break
    cv2.destroyAllWindows()
