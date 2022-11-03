import cv2
import mediapipe as mp
import pandas as pd
import os

landmarks_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

# -1. Function
def make_landmark(results):
    cur_lm = []
    for id,lm in enumerate(results.pose_landmarks.landmark):
        # if id <= 10 or 17<=id<=22 or id >= 29: 
        #     continue

        cur_lm.append(lm.x)
        cur_lm.append(lm.y)
        cur_lm.append(lm.z)
        # cur_lm.append(lm.visibility)
    return cur_lm

# 0. Initialization
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

videos_names = os.listdir('Data Video\Data Train\Squat')
for video_name in videos_names:
    lm_list = list()
    
    if 'mp4' not in video_name or 'right' in video_name:
        continue

    label = video_name[:video_name.find('.')]

    # 1. Load video from dataset
    vid = cv2.VideoCapture(f'Data Video\Data Train\Squat\{label}.mp4')

    # 2. Use mediapipe to detect 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret,frame = vid.read()

            if not ret:
                break
            
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Record key point
                lm = make_landmark(results)

                lm_list.append(lm)
                # Draw on frame
                mp_draw.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(0,0,255),thickness = 2, circle_radius = 2),
                                mp_draw.DrawingSpec(color=(255,255,255),thickness = 2, circle_radius = 2)
                                )
            cv2.imshow(label,image)

            if cv2.waitKey(1) == 27:
                break

    vid.release()
    cv2.destroyAllWindows()

    # 3. Save data to file 
    df = pd.DataFrame(lm_list)

    df.to_csv(f'Data TXT/Squat/{label}.txt')