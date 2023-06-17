import cv2

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():
    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord("s"):  # wait for 's' key to save and exit
        path = r"C:\SEM6\PCPO\cameraCalibration\images"
        path += r"\img" + str(num) + ".png"
        cv2.imwrite(path, img)
        print("image saved!", path)
        num += 1

    cv2.imshow("Img", img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
