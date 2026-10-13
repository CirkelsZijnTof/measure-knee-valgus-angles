# ReKnee prototype for the following purposes:
# - 

import mediapipe as mp
import numpy as np
import pandas as pd
import copy
import cv2
import time
import datetime
import os

class ReKnee_utils:

    def __init__(self):

        self.default_error_code = 0

        self.error_codes = {
            0: 'measurement successful',
            1: 'measurement skipped: subject out of view',
            2: 'measurement skipped: low visibility',
            3: 'execution halted: missing filepath to analysis file'
        }
        
        self.dct_pnt2idx = {
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

        self.dct_idx2pnt = {
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
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index" 
        }
    
    @staticmethod
    def vector(p1, p2): 

        v = []
        for w1, w2 in zip(p1, p2):
            v.append(w2 - w1)

        return np.array(v)

    @staticmethod
    def amplitude(vector):

        vector_amplitude = (vector**2).sum()**0.5
        
        return vector_amplitude
    
    # calculates angle between two vectors (degrees)
    @staticmethod
    def get_angle(vector_1, vector_2):

        dot_product = np.dot(vector_1, vector_2)

        cos_theta = dot_product / (ReKnee_utils.amplitude(vector_1) * ReKnee_utils.amplitude(vector_2))
        theta = np.arccos(cos_theta)

        theta = np.rad2deg(theta)
        
        return round(theta, 2)
    
    @staticmethod
    def get_angle_from_points(p_centre, p1, p2):

        v1 = ReKnee_utils.vector(p_centre, p1)
        v2 = ReKnee_utils.vector(p_centre, p2)

        theta = ReKnee_utils.get_angle(v1, v2)

        return theta

    @staticmethod
    def get_timestamp(curr_time=None,format='%Y-%m-%d_%H:%M:%S.%f'):

        if curr_time is not None:
            return datetime.datetime.fromtimestamp(curr_time).strftime(format)
        else:
            return datetime.datetime.fromtimestamp(time.time()).strftime(format)

    @staticmethod
    def pause(RK_outp, waittime):

        default_frame = np.zeros(shape=RK_outp.RK_input.feed_dims, dtype=np.uint8)

        success, frame = RK_outp.RK_input.feed.read()

        init_time = time.time()
        curr_time = 0
        end_time = init_time + waittime

        font = cv2.FONT_HERSHEY_SIMPLEX

        dims    = frame.shape
        height  = dims[0] # x
        width   = dims[1] # y

        txtposition = (int(width / 2), int(height / 3))

        key = cv2.waitKey(1)

        while end_time > curr_time:

            success, frame = RK_outp.RK_input.feed.read()

            curr_time = time.time()
            time_left = end_time - curr_time

            # clamp time_left to prevent negatives
            if time_left < 0:
                time_left = 0.0

            # FIX: HIDE FEED DOES NOT ANONIMIZE
            if RK_outp.do_anonimize:
                frame = copy.copy(default_frame)

            cv2.putText(frame, f'{round(time_left,2)}', txtposition, font, 3, (50,80,240), 5, cv2.LINE_4)
            cv2.imshow("Pose Detection", frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

# 2nd fase: input
class ReKnee_input:

    def __init__(self, source=None, hide_feed=False, static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                 enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.hide_feed = hide_feed
        self.key = None
        self.amount_of_frames = None
        self.video_as_source = False

        if source is not None:
            self.feed = cv2.VideoCapture(source)
            self.amount_of_frames = self.feed.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_as_source = True

        else:
            self.feed = cv2.VideoCapture(0) # trigger for webcam

        # probe feed
        success, raw_frame = self.feed.read()
        self.feed_dims = raw_frame.shape

        # initialise mediapipe solutions
        self.mp_pose = mp.solutions.pose
        
        # specify mediapipe settings
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def get_key_and_wait(self):

        if not self.hide_feed:
            self.key = cv2.waitKey(1)

        else:
            pass

# 2nd fase: processing
class ReKnee_processing:
    
    def __init__(self, RK_input, visibility_threshold=0.8, filter_on_primary=True):
        
        # utils
        self.utils = ReKnee_utils()

        # internal error handling for processing
        self.init_time = time.time()
        self.curr_error_code = None

        # get data from input class
        self.RK_input = RK_input

        # handle videofeed
        self.frame_number = 1
        self.success = False
        self.raw_frame = None
        self.rgb_frame = None

        # intended output columns
        self.columns = ['timestamp','delta_t','index','landmark','visibility','x','y','z','delta_x','delta_y','delta_z','left_knee_valgus','right_knee_valgus']
        self.angle_landmarks = {'left_knee_valgus': ['left_knee','left_ankle','left_hip'],
                                'right_knee_valgus': ['right_knee','right_ankle','right_hip'],}

        # main output landmarks --> controls what landmarks cause failure to measure
        self.vis_threshold = visibility_threshold
        self.filter_on_primary = filter_on_primary
        self.primary_column = 'landmark'
        self.primary_landmarks = ['left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot_index','right_foot_index']

        # initiate time stuff
        self.prev_time = None
        self.curr_time = 0
        self.delta_time = 0

        # initiate landmark stuff
        self.prev_landmarks = None
        self.curr_landmarks = None

        self.prev_x = 0
        self.prev_y = 0
        self.prev_z = 0

        self.curr_x = 0
        self.curr_y = 0
        self.curr_z = 0

        self.dx = 0
        self.dy = 0
        self.dz = 0

        self.left_knee_valgus = 0
        self.right_knee_valgus = 0

        self.prev_result = None
        self.curr_result = None

    def print_error_status(self):
        
        running_time = self.curr_time - self.init_time

        # os.system('cls')
        if self.RK_input.video_as_source:
            percentage_done = round(self.frame_number / self.RK_input.amount_of_frames * 100, 2)
            print(f'[{round(running_time, 2)}s] {self.utils.error_codes[self.curr_error_code]} ({percentage_done}% done)')

        else:
            print(f'[{round(running_time, 2)}s] {self.utils.error_codes[self.curr_error_code]}')

        self.curr_error_code = int(self.utils.default_error_code)

    # update internal time: execute every frame
    def update_time(self):

        self.curr_time = time.time()

        if self.prev_time is not None:
            self.delta_time = self.curr_time - self.prev_time

        self.prev_time = float(self.curr_time)

    # pre-read before markerless mocap to quit execution early
    def read_frame(self):

        # read frame from feed
        self.success, self.raw_frame = self.RK_input.feed.read()
        self.frame_number += 1

    # execute markerless mocap and store data
    def get_landmarks(self):
        
        # cv2 frames come in BGR format. Change to RGB: 
        self.rgb_frame = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)

        self.pose_pred = self.RK_input.pose.process(self.rgb_frame)

        # check for succesful read of pose in frame
        if self.pose_pred.pose_landmarks is not None:

            self.curr_landmarks = self.pose_pred.pose_landmarks.landmark
            
            self.curr_result = pd.DataFrame(columns=self.columns)

            for idx, lm in enumerate(self.curr_landmarks):
                
                timestamp = self.utils.get_timestamp(self.curr_time)

                self.curr_x = self.curr_landmarks[idx].x
                self.curr_y = self.curr_landmarks[idx].y
                self.curr_z = self.curr_landmarks[idx].z

                if self.prev_landmarks is not None:

                    self.prev_x = self.prev_landmarks[idx].x
                    self.prev_y = self.prev_landmarks[idx].y
                    self.prev_z = self.prev_landmarks[idx].z

                dx = self.curr_x - self.prev_x
                dy = self.curr_y - self.prev_y
                dz = self.curr_z - self.prev_z

                curr_landmark = self.utils.dct_idx2pnt[idx]
                curr_visibility = round(lm.visibility, 3)

                temp_row = pd.Series([timestamp, self.delta_time, idx, curr_landmark, curr_visibility, self.curr_x, self.curr_y, self.curr_z, 
                                      dx, dy, dz, self.left_knee_valgus, self.right_knee_valgus], index=self.columns)
                temp_row = temp_row.to_frame().T

                # add temp row to bottom of current result
                self.curr_result = pd.concat([self.curr_result, temp_row], axis=0, join='inner')

            # calculate and add knee valgus angles -> fill entire column
            for key, lm in self.angle_landmarks.items():

                p_0 = (self.curr_result[['x','y']].loc[self.curr_result['landmark'].isin([lm[0]])].values)[0]
                p_1 = (self.curr_result[['x','y']].loc[self.curr_result['landmark'].isin([lm[1]])].values)[0]
                p_2 = (self.curr_result[['x','y']].loc[self.curr_result['landmark'].isin([lm[2]])].values)[0]

                self.curr_result[key] = 180 - self.utils.get_angle_from_points(p_0, p_1, p_2)

            # calculate average visibility
            if self.filter_on_primary:
                avg_vis = np.mean(self.curr_result[self.curr_result[self.primary_column].isin(self.primary_landmarks)]['visibility'])
            else:
                avg_vis = np.mean(self.curr_result['visibility'])

            # store current landmarks to be saved.
            if avg_vis >= self.vis_threshold:
                self.prev_landmarks = list(self.curr_landmarks)
            else:
                self.curr_error_code = 2

        else:
            self.curr_error_code = 1

        # self.print_error_status()

        print(self.curr_result['left_knee_valgus'].iloc[0], self.curr_result['right_knee_valgus'].iloc[0])

        return self.curr_result
        
# 2nd fase: interim output
# various functions for outputting landmark data, before moving on to fase 3
class ReKnee_output:

    def __init__(self, RK_input, RK_proc, anonimize=True, do_draw_skeleton=True, filepath='ReKnee measurement ', add_timestamp=True): 
        
        self.RK_input = RK_input
        self.RK_proc = RK_proc
        self.do_draw_skeleton = do_draw_skeleton
        self.do_anonimize = anonimize

        self.filepath = filepath

        self.init_timestamp = ReKnee_utils.get_timestamp(format='%Y-%m-%d_%H-%M-%S')

        if add_timestamp:
            self.filepath += self.init_timestamp
        self.filepath += '.csv'

        self.mp_drawing = mp.solutions.drawing_utils

        self.default_frame = np.zeros(shape=self.RK_input.feed_dims, dtype=np.uint8) # black frame of size feed dims

    def display_frame(self): 
        
        if not self.RK_input.hide_feed:

            if self.do_anonimize:
                frame = copy.copy(self.default_frame) # replaces frame with default black frame

            else:
                frame = self.RK_proc.raw_frame

            if self.do_draw_skeleton:

                self.mp_drawing.draw_landmarks(
                    frame,
                    self.RK_proc.pose_pred.pose_landmarks,
                    self.RK_input.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
            
            cv2.imshow("Pose Detection", frame)

        else:
            pass

    def save_results(self):

        if self.RK_proc.prev_result is not None:
            self.RK_proc.prev_result.to_csv(self.filepath, index=False, header=False, mode='a')

# for running knee instability assessment.
class ReKnee_test: 

    def __init__(self, RK_proc, RK_output=None, filepath=None): 
        
        self.utils = ReKnee_utils()
        self.process_new_file = False

        self.RK_proc = RK_proc

        if filepath is not None:
            self.filepath = filepath

        elif RK_output is not None:
            self.filepath = RK_output.filepath
            self.process_new_file = True

        else:
            raise Exception(self.utils.error_codes[3])
        
        self.sep = ','
        self.data = None

    def read_data(self):

        # read from assigned .csv file
        self.data = pd.read_csv(self.filepath, sep=self.sep)
        self.data.columns = self.RK_proc.columns # re-add column header

    def print_results(self):

        # os.system('cls')

        print('results.')
        print('advice for prevention or something if there is reason to worry.\n')
        print('disclaimer that prototype results are not validated to be taken as ')
        print('medical advice and to always consult your GP if you are still worried.\n')

    def run_sequence(self):

        self.read_data() # ready data for analysis

        ... # knee wobble index or whatever

        print('executed test sequence!')

        self.print_results()
        

