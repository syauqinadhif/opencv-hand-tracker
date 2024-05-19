import cv2
import numpy as np
import HandTrackerModule as htm

brushThickness = 15

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetectors(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # read webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find hand landmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        xp, yp = 0, 0
        # print(lmList)

        # tip of index finger
        x1, y1 = lmList[8][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1]:
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            print("Painting Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), (0, 0, 255), brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1),
                     (0, 0, 255), brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Painter", img)
    cv2.waitKey(1)
