# code to display markerless motion capture.
# use mediapipe: commonly used in literature for simple 2d use-cases.
# our use is a 2d use-case, reasonably accurate for 2d stuff.

import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import time
import datetime
import copy

def get_timestamp(curr_time=None,format='%Y-%m-%d_%H:%M:%S.%f'):

    if curr_time is not None:
        return datetime.datetime.fromtimestamp(curr_time).strftime(format)
    else:
        return datetime.datetime.fromtimestamp(time.time()).strftime(format)

# parts of body and corresponding landmark index
DICT_POINT2IDX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

DICT_IDX2POINT = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18:"right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26:"right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index" 
}

columns = ['timestamp','delta_t','index','landmark','visibility','x','y','z','delta_x','delta_y','delta_z']

separator = ','

on = True

# initialise mediapipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# specify mediapipe settings
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture('V_0.mp4') # or 0

initial_timestamp = get_timestamp()
file_name = 'ReKnee measurement ' + get_timestamp(format='%Y-%m-%d_%H-%M-%S') + '.csv'

prev_landmarks = None
curr_landmarks = None

curr_time = 0
prev_time = 0
delta_time = 0

while on:

    success, frame = cap.read()
    current_time = get_timestamp()

    curr_time = time.time()
    delta_time = get_timestamp(curr_time=(curr_time - prev_time), format='%S.%f')
    prev_time = float(curr_time)

    if not success:
        print('Program quit unexpectedly due to unsuccesful camera read at ', current_time)
        print('Please check if a webcam is connected and working.')

        break

    # convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # run pose detection
    result = pose.process(rgb_frame)

    if result.pose_landmarks is not None:

        curr_landmarks = result.pose_landmarks.landmark

        for idx, lm in enumerate(curr_landmarks):

            if prev_landmarks is not None:

                dx = curr_landmarks[idx].x - prev_landmarks[idx].x
                dy = curr_landmarks[idx].y - prev_landmarks[idx].y
                dz = curr_landmarks[idx].z - prev_landmarks[idx].z

            else:

                dx = curr_landmarks[idx].x - 0
                dy = curr_landmarks[idx].y - 0
                dz = curr_landmarks[idx].z - 0

            #FIX THIS SO DELTA W CAN BE TAKEN INTO ACCOUNT
            landmark_result = pd.Series([current_time, delta_time, idx, DICT_IDX2POINT[idx], round(lm.visibility,3), lm.x, lm.y, lm.z, dx, dy, dz], index=columns)
            landmark_result = landmark_result.to_frame().T

            # save current data to .csv file
            landmark_result.to_csv(file_name, index=False, header=False, mode='a')

            # print(f"Landmark {idx}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}, vis={lm.visibility:.4f}")

        prev_landmarks = list(curr_landmarks)

        # draw skeleton on the frame
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

    else:
        # Print all landmark positions
        print('No measurement was taken at ', current_time)

    # display frame
    cv2.imshow("Pose Detection", frame)
    
    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        print('Program was quit normally at ', current_time)

        break