# ReKnee prototype for the following purposes:
# - 

import mediapipe as mp
import numpy as np
import pandas as pd
import copy
import cv2
import time
import datetime
import matplotlib.pyplot as plt
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
    def strip_filename(filename, mode='no_suffix'):

        if mode == 'no_suffix':
            position = 0
        else:
            position = 1

        return os.path.splitext(filename)[position]

    @staticmethod
    def pause(RK_proc, RK_outp, waittime):

        default_frame = np.zeros(shape=RK_outp.RK_input.feed_dims, dtype=np.uint8)

        success, frame = RK_outp.RK_input.feed.read()
        RK_proc.frame_number += 1

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
            RK_proc.frame_number += 1

            print('Processing paused!')

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

    def __init__(self, source=None, hide_feed=False, focus='left', static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                 enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.source_name = source
        self.hide_feed = hide_feed
        self.focus = focus

        self.key = None
        self.amount_of_frames = None
        self.video_as_source = False

        if source is not None and type(source) == str:
            self.feed = cv2.VideoCapture(source)

            self.amount_of_frames = self.feed.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_as_source = True
            self.fps = self.feed.get(cv2.CAP_PROP_FPS)

        else:
            self.feed = cv2.VideoCapture(0) # trigger for webcam

            self.fps = 15

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
    
    def __init__(self, RK_input, visibility_threshold=0.8, filter_on_primary=True, analysis_axes=['x','y']):
        
        # utils
        self.utils = ReKnee_utils()

        # internal error handling for processing
        self.init_time = time.time()
        self.curr_error_code = 0

        # get data from input class
        self.RK_input = RK_input

        # handle videofeed
        self.frame_number = 1
        self.success = False
        self.raw_frame = None
        self.rgb_frame = None

        # intended output columns
        self.data = {'general': {'columns': ['timestamp','delta_t','index','landmark','visibility','x','y','z','delta_x','delta_y','delta_z'], 'data': None},
                    'valgus': {'columns': ['timestamp','left_knee_valgus','right_knee_valgus', 'v_y_core'], 'data': None}}

        self.valgus_landmarks = {'left_knee_valgus': ['left_knee','left_ankle','left_hip'],
                                'right_knee_valgus': ['right_knee','right_ankle','right_hip'],}

        # main output landmarks --> controls what landmarks cause failure to measure
        self.vis_threshold = visibility_threshold
        self.filter_on_primary = filter_on_primary
        self.primary_column = 'landmark'
        self.primary_landmarks = ['left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

        # initiate time stuff
        self.prev_time = None
        self.curr_time = 0
        self.delta_time = 1

        # initiate landmark stuff
        self.general_prev_landmarks = None
        self.general_curr_landmarks = None

        self.prev_x = 0
        self.prev_y = 0
        self.prev_z = 0

        self.curr_x = 0
        self.curr_y = 0
        self.curr_z = 0

        self.dx = 0
        self.dy = 0
        self.dz = 0

        self.default_valgus = 0

        self.analysis_axes = analysis_axes

        self.general_prev_result = None
        self.general_curr_result = None

        self.valgus_curr_result = None
        self.valgus_prev_result = None

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

            if self.RK_input.video_as_source:
                self.delta_time = 1 / self.RK_input.fps

            else:
                self.delta_time = self.curr_time - self.prev_time

        self.prev_time = float(self.curr_time)

    # pre-read before markerless mocap to quit execution early
    def read_frame(self):

        # read frame from feed
        self.success, self.raw_frame = self.RK_input.feed.read()
        self.frame_number += 1

    def fill_curr_general(self, timestamp):

        # iterate through current found landmarks
        for idx, lm in enumerate(self.general_curr_landmarks):
            
            # get current coordinates of lm
            self.curr_x = self.general_curr_landmarks[idx].x
            self.curr_y = self.general_curr_landmarks[idx].y
            self.curr_z = self.general_curr_landmarks[idx].z

            # get previous coordinates of lm
            if self.general_prev_landmarks is not None:

                self.prev_x = self.general_prev_landmarks[idx].x
                self.prev_y = self.general_prev_landmarks[idx].y
                self.prev_z = self.general_prev_landmarks[idx].z

            # calculate difference between previous and current coordinates for lm
            dx = self.curr_x - self.prev_x
            dy = self.curr_y - self.prev_y
            dz = self.curr_z - self.prev_z

            curr_landmark = self.utils.dct_idx2pnt[idx]
            curr_visibility = round(lm.visibility, 3)

            temp_row = pd.Series([timestamp, self.delta_time, idx, curr_landmark, curr_visibility, self.curr_x, self.curr_y, self.curr_z, 
                                    dx, dy, dz], index=self.data['general']['columns'])
            temp_row = temp_row.to_frame().T

            # add temp row to bottom of current result
            self.general_curr_result = pd.concat([self.general_curr_result, temp_row], axis=0, join='inner')

    def fill_curr_valgus(self, timestamp):

        # this program considers pelvis as middle of body
        # pelvis is between the hips. average between height position is used to approximate height of middle
        
        # difference between height previous and current, calculated earlier
        dx_left_hip = self.general_curr_result['delta_y'].loc[self.general_curr_result['landmark'].isin(['left_hip'])].values[0]
        dx_right_hip = self.general_curr_result['delta_y'].loc[self.general_curr_result['landmark'].isin(['right_hip'])].values[0]

        # difference between heightward velocity previous and current, divide by delta_t 
        dv_left_hip = dx_left_hip / self.delta_time
        dv_right_hip = dx_right_hip / self.delta_time

        # average
        speed_core = -np.mean([dv_left_hip, dv_right_hip]) * 100 # multiply by 100, converting cm to m

        # prepare valgus table row --> timestamp & pelvis velocity already added. valgus is rubbish to be overwritten
        temp_row = pd.Series([timestamp, self.default_valgus, self.default_valgus, speed_core], index=self.data['valgus']['columns'])

        # calculate and add knee valgus angles -> fill entire column
        for key, lm in self.valgus_landmarks.items():

            p_0 = self.general_curr_result[self.analysis_axes].loc[self.general_curr_result['landmark'].isin([lm[0]])].values[0]
            p_1 = self.general_curr_result[self.analysis_axes].loc[self.general_curr_result['landmark'].isin([lm[1]])].values[0]
            p_2 = self.general_curr_result[self.analysis_axes].loc[self.general_curr_result['landmark'].isin([lm[2]])].values[0]

            temp_row[key] = 180 - self.utils.get_angle_from_points(p_0, p_1, p_2)

        self.valgus_curr_result = temp_row.to_frame().T

    # execute markerless mocap and store data
    def get_data(self):
        
        # cv2 frames come in BGR format. Change to RGB: 
        self.rgb_frame = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)

        self.pose_pred = self.RK_input.pose.process(self.rgb_frame)

        # check for succesful read of pose in frame
        if self.pose_pred.pose_landmarks is not None:
            
            # estimate landmarks
            # generally an estimate of distance in cm.
            self.general_curr_landmarks = self.pose_pred.pose_landmarks.landmark
            
            # update timestamp
            timestamp = self.utils.get_timestamp(self.curr_time)

            # --- two tables are being maintained each loop: general and valgus ---
            # fill general table
            self.general_curr_result = pd.DataFrame(columns=self.data['general']['columns'])
            self.fill_curr_general(timestamp)

            # fill valgus table
            self.valgus_curr_result = pd.DataFrame(columns=self.data['valgus']['columns'])
            self.fill_curr_valgus(timestamp)

            # calculate average visibility
            if self.filter_on_primary:
                avg_vis = np.mean(self.general_curr_result[self.general_curr_result[self.primary_column].isin(self.primary_landmarks)]['visibility'])
            else:
                avg_vis = np.mean(self.general_curr_result['visibility'])

            # store current landmarks to be saved.
            if avg_vis >= self.vis_threshold:
                self.general_prev_landmarks = list(self.general_curr_landmarks)
            else:
                self.curr_error_code = 2

        else:
            self.curr_error_code = 1

        self.print_error_status()

        if self.curr_error_code == 0: # error_code 0 means succesful read

            self.data['general']['data'] = self.general_curr_result
            self.data['valgus']['data'] = self.valgus_curr_result

            # # uncomment below to reveal current knee valgus states
            # print(self.general_curr_result['left_knee_valgus'].iloc[0], self.general_curr_result['right_knee_valgus'].iloc[0])

            self.general_prev_result = copy.copy(self.general_curr_result)
            self.valgus_prev_result = copy.copy(self.valgus_curr_result)

        return self.data # for more expandable code: work with this outside of class
        
# 2nd fase: interim output
# various functions for outputting landmark data, before moving on to fase 3
class ReKnee_output:

    def __init__(self, RK_input, RK_proc, anonimize=True, do_draw_skeleton=True, valgus_text_on_frame=True, filepath='ReKnee measurement ', add_timestamp=False): 
        
        self.RK_input = RK_input
        self.RK_proc = RK_proc

        padding = 50
        self.txt_position_1 = (padding, padding)
        self.txt_position_2 = (self.txt_position_1[0], self.txt_position_1[1]+padding)
        self.txt_colour = (50,80,240)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.valgus_text_on_frame = valgus_text_on_frame
        self.do_draw_skeleton = do_draw_skeleton
        self.do_anonimize = anonimize

        self.filepath = filepath

        self.init_timestamp = ReKnee_utils.get_timestamp(format='%Y-%m-%d_%H-%M-%S')

        if add_timestamp:
            self.filepath += self.init_timestamp
        elif RK_input.video_as_source:
            self.filepath += ReKnee_utils.strip_filename(RK_input.source_name)
        self.filepath += '.csv'

        self.mp_drawing = mp.solutions.drawing_utils

        self.default_frame = np.zeros(shape=self.RK_input.feed_dims, dtype=np.uint8) # black frame of size feed dims

    def display_frame(self): 
        
        # do not show a window if feed is hidden
        if not self.RK_input.hide_feed:

            if self.do_anonimize:
                frame = copy.copy(self.default_frame) # anonimize

            else:
                frame = self.RK_proc.raw_frame # use real video as background

            if self.do_draw_skeleton:
                
                # overlay landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    self.RK_proc.pose_pred.pose_landmarks,
                    self.RK_input.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

            if self.valgus_text_on_frame:
                
                # overlay valgus results per frame
                left_valgus = round(self.RK_proc.data['valgus']['data']['left_knee_valgus'].values[0], 2)
                right_valgus = round(self.RK_proc.data['valgus']['data']['right_knee_valgus'].values[0], 2)

                cv2.putText(frame, f'left: {left_valgus} deg', self.txt_position_1, self.font, 1, self.txt_colour, 1, cv2.LINE_4)
                cv2.putText(frame, f'right: {right_valgus} deg', self.txt_position_2, self.font, 1, self.txt_colour, 1, cv2.LINE_4)

            cv2.imshow("Pose Detection", frame)

        else:
            pass

    def save_results(self):

        output_data = self.RK_proc.data

        # iterate through keys in data
        for key in output_data.keys():

            curr_data = output_data[key]['data']
            
            if curr_data is not None:

                curr_data.to_csv(key + ' ' + os.path.basename(self.filepath), index=False, header=False, mode='a')

class ReKnee_visualise:

    def __init__(self, RK_input, RK_proc, data): 
        
        self.RK_inpt = RK_input
        self.RK_proc = RK_proc

        self.data = data

    def graph_valgus(self, deg_columns, v_columns=None, threshold=None):

        # plot settings
        str_title = ReKnee_utils.strip_filename(self.RK_inpt.source_name) if self.RK_inpt.video_as_source else 'Webcam footage'
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # --- top ---
        ax1.plot(self.data.index, self.data[deg_columns])
        if threshold is not None:
            ax1.axhline(y=threshold, color='r', linestyle=':')

        ax1.set_ylim(0,155)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Degrees')
        ax1.set_title(f'{str_title}')
        ax1.legend(deg_columns, loc='upper right')
        
        # --- bottom ---
        if v_columns is not None:
            
            ax2.plot(self.data.index, self.data[v_columns])
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Meters per second')
            ax2.set_title('Core velocity')
            ax2.legend(v_columns, loc='upper right')

        plt.tight_layout()
        plt.show()

# for running knee instability assessment after input data was fully analyzed.
class ReKnee_test: 

    def __init__(self, RK_inpt, RK_proc, RK_output=None, filepath=None, do_visual=True): 
        
        self.utils = ReKnee_utils()
        self.process_new_file = False

        self.RK_inpt = RK_inpt
        self.RK_proc = RK_proc

        if filepath is not None:
            self.filepath = filepath
        elif RK_output is not None:
            self.filepath = RK_output.filepath
            self.process_new_file = True
        else:
            # halt program execution if there is or will be no file to work with!
            raise Exception(self.utils.error_codes[3])
        
        self.sep = ','
        self.valgus_data = {'columns': self.RK_proc.data['valgus']['columns'],'valgus_columns': ['left_knee_valgus','right_knee_valgus'], 'data': None, 'results': {'left_knee_valgus': 0, 'right_knee_valgus': 0}} # <-- change later to make more dependent on RK_proc input

        self.do_visual = do_visual

        self.pose_error = 0.9
        self.incl_landmarks = len(self.RK_proc.primary_landmarks) // 2
        self.program_error = self.pose_error ** self.incl_landmarks

        self.allowance = 1+(1-self.program_error)
        self.min_knee_instability_threshold_angle = 5
        self.max_knee_instability_threshold_angle = 10
        
        self.min_threshold_angle = self.min_knee_instability_threshold_angle * self.allowance
        self.max_threshold_angle = self.max_knee_instability_threshold_angle * self.allowance

        self.focus_result = 0

        if self.process_new_file:
            self.delete_preexisting_data()

    def print_prevention_advice(self):
        
        print('Follow the below tips to prevent ACL-injuries:')
        
        print('1| Jump, land, stop and move with your knees directly over your feet.')
        print('2| Never let your knees collapse inward (high valgus angle).')
        print('3| Strengthen your hips and thighs through squats or lunges or targeted exercise machines. Quality > quantity.')
        print('4| Practice proper balance with various ball exercises such as juggling and the single leg ball pass.')
        print('5| Be sure to warm up and stretch before games and practice!\n')
        
    def print_header(self):

        print('\nGenerally, a <5-10 degrees valgus angle is considered indicative of a stable knee.')
        print(f'You have selected that the video shows you squatting using your {self.RK_inpt.focus} leg.\n')

    def print_major_risk(self):

        print('\nUnfortunately, we have detected a major risk of suffering an ACL injury.')
        print(f'You had an average valgus angle larger than {self.max_knee_instability_threshold_angle} degrees.\n')

    def print_minor_risk(self):

        print('\nUnfortunately, we have detected a minor risk of suffering an ACL injury.')
        print(f'You had an average valgus angle larger than {self.min_knee_instability_threshold_angle} degrees.')
        print('Please note that this is within the healthy valgus range and there is not cause for immediate concern.\n')

    def print_little_risk(self):

        print('\nBased on the gathered data, ReKnee has not detected a significantly increased risk of suffering an ACL injury.\n')

    def print_disclaimer(self):

        print('Disclaimer:')
        print('ReKnee results are not to be taken as medical advice!')
        print('Always consult your GP if you are (still) worried.\n')

    def delete_preexisting_data(self):

        output_data = self.RK_proc.data

        for key in output_data.keys():

            curr_filename = key+' '+self.filepath

            if os.path.exists(curr_filename):

                os.remove(curr_filename)

    def read_data(self):

        # read from assigned .csv file
        if self.process_new_file:
            self.filepath = 'valgus '+self.filepath

        self.valgus_data['data'] = pd.read_csv(self.filepath, sep=self.sep) # <-- change later to not be hardcoded
        self.valgus_data['data'].columns = self.valgus_data['columns'] # re-add column header to general dataframe

    def generate_results(self):

        result = float(np.mean(self.valgus_data['data'][self.RK_inpt.focus+'_knee_valgus']))
        self.focus_result = round(result, 2)

        for col in self.valgus_data['valgus_columns']:

            result = float(np.mean(self.valgus_data['data'][col]))
            self.valgus_data['results'][col] = round(result, 2)

    def print_results(self):

        # os.system('cls')
        self.print_header()

        for key, result in self.valgus_data['results'].items():
            print(f'{key}:\t{result}')

        if self.focus_result >= self.max_threshold_angle:
            self.print_major_risk()

        elif self.focus_result >= self.min_threshold_angle:
            self.print_minor_risk()
        
        else:
            self.print_little_risk()
        
        self.print_prevention_advice()
        self.print_disclaimer()

        if self.do_visual:
            ReKnee_visualise(RK_input=self.RK_inpt, RK_proc=self.RK_proc, data=self.valgus_data['data']).graph_valgus(self.valgus_data['valgus_columns'], v_columns='v_y_core', threshold=10)

    def run_sequence(self):

        self.read_data() # ready data for analysis

        self.generate_results()

        self.print_results()
        

