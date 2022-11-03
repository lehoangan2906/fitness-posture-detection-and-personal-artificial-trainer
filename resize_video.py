import mediapipe
import numpy as np
import pandas as pd
import cv2
import os 

videos_names = os.listdir('Data Video\Data Train\Squat\Original')
for video_name in videos_names:
    if 'mp4' not in video_name:
        continue

    label = video_name[:video_name.find('.')]
    new_label = video_name[:video_name.find('.')]

    vid = cv2.VideoCapture(f'Data Video\Data Train\Squat\Original\{label}.mp4')

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    fps = vid.get(5)
    size = (frame_width, frame_height)

    new_vid = cv2.VideoWriter(f'Data Video\Data Train\Squat\{new_label}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (576,320))

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        new_frame = cv2.resize(frame, (576,320))

        new_vid.write(new_frame)

        cv2.imshow(label,new_frame)

        if cv2.waitKey(30) == 27:
            break

    vid.release()
    new_vid.release()

    cv2.destroyAllWindows()