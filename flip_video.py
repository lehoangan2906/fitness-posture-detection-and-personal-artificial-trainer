import mediapipe
import numpy as np
import pandas as pd
import cv2
import os

label = 'T_situp_1'
new_label = 'T_situp_left_1'

vid = cv2.VideoCapture(f'Data Video\Data Test\{label}.mp4')

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = vid.get(5)
size = (frame_width, frame_height)

new_vid = cv2.VideoWriter(f'Data Video\Data Test\{new_label}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (576,320))

while True:
    ret, frame = vid.read()

    if not ret:
        break

    new_frame = cv2.flip(frame, 1)

    new_vid.write(new_frame)

    cv2.imshow(new_label,new_frame)

    if cv2.waitKey(1) == 27:
        break

vid.release()
new_vid.release()

cv2.destroyAllWindows()