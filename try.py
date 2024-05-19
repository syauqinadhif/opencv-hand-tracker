import cv2
import mediapipe as mp
import HandTrackerModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetectors()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    # if len(lmList) != 0:
    #     print(lmList[0])

    cv2.imshow("Video Stream", img)
    cv2.waitKey(1)