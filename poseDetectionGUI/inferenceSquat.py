from tabnanny import verbose
import venv
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

label = "Warmup...."

model_squat = "model_squat_fullLm.h5"



no_of_timesep_squat = 60
    
def make_landmark_timestep(results):
    cur_lm = []
    for id,lm in enumerate(results.pose_landmarks.landmark):

        cur_lm.append(lm.x)
        cur_lm.append(lm.y)
        cur_lm.append(lm.z)
        # cur_lm.append(lm.visibility)
    return cur_lm

def draw_landmark_on_image(mpDraw, results, img,mpPose):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

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

def draw_class_on_image(label,counter, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    strCounter = "Count: "+ str(counter)
    bottomLeftCornerOfText = (10, 60)
    cv2.putText(img, strCounter,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list
                            ,verbose=0)
    print(np.argmax(results))
    
    if np.amax(results) <=0.75:
        label ="squat di"
    elif(np.argmax(results) == 0):
        label = "Dung Form"
    elif(np.argmax(results)==1):
        label = "Sai Form chum chan"
    else:
        (np.argmax(results)==2)
        label = "Sai Form cong lung"

    
    return label

def calculate_angle(landmarks,mppose):

    a = np.array([landmarks[mppose.PoseLandmark.LEFT_HIP.value].x,landmarks[mppose.PoseLandmark.LEFT_HIP.value].y]) # First
    b = np.array([landmarks[mppose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mppose.PoseLandmark.LEFT_KNEE.value].y]) # Mid
    c = np.array([landmarks[mppose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mppose.PoseLandmark.LEFT_ANKLE.value].y]) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle1 = np.abs(radians*180.0/np.pi)
    if angle1 >180.0:
        angle1 = 360- angle1
    return angle1

        

def squatDetect():
    
    n_time_steps = no_of_timesep_squat

    lm_list = []

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    mpDraw = mp.solutions.drawing_utils

    model = tf.keras.models.load_model(model_squat)

    # cap = cv2.VideoCapture(3)
    cap = cv2.VideoCapture("C:\code\AI\pose_detection\data_test\T_push_6.mp4")
    i = 0
    warmup_frames = 1
    counter = 0 
    stage = None
    angle = 0
    fps = 0
    while True:

        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i = i + 1
        check = False
        if i > warmup_frames:
            global label 
            
            try:
                
                landmarks = results.pose_landmarks.landmark
                check = True
            except:
                check = False
                label = "Not Found body"
            if check == True:
                angle = calculate_angle(landmarks,mpPose) 
                
                cv2.putText(img, str(angle), 
                            tuple(np.multiply([landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x,landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y], [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                
                # print(angle)
                if results.pose_landmarks:
                    c_lm = make_landmark_timestep(results)

                    lm_list.append(c_lm)
                    if len(lm_list) == n_time_steps:
                        t1 = threading.Thread(target=detect, args=(model, lm_list,))
                        t1.start()
                        lm_list = []

                    img = draw_landmark_on_image(mpDraw, results, img,mpPose)
        if label == "Dung Form":
            if angle <130:
                stage ="down"
            if angle> 150 and stage =='down':
                stage="up"
                counter +=1
            
                
        img = draw_class_on_image(label,counter, img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


        
        
