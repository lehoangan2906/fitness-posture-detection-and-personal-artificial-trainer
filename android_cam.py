import cv2

url = 'http://192.168.137.177:4747/video'

cam = cv2.VideoCapture(url)

while True:
    ret, frame = cam.read()

    if not ret:
        break

    cv2.imshow('Android Cam', frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()