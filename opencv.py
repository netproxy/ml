import cv2
import time
cap = cv2.VideoCapture(1)

time.sleep(2)
while True:
    ret, im = cap.read()
    if ret:
        cv2.imshow('video test', im)
        key = cv2.waitKey(27)
        if key == 27:

            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
