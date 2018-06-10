import cv2
capture = cv2.VideoCapture(0)
while True:
    ret, img = capture.read()
    if img is not None:
        cv2.imshow('camera', img)
    if cv2.waitKey(10) & 0xFF == ord(' '):
        break

capture.release()
cv2.destroyAllWindows()