import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
from collections import deque


class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed = 0

    def start(self):
        """Start or resume the timer."""
        if self._start_time is None:
            self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer and calculate elapsed time."""
        if self._start_time is not None:
            self._elapsed += (time.perf_counter() - self._start_time) * 1000  # in milliseconds
            self._start_time = None

    def reset(self):
        """Reset the timer."""
        self._start_time = None
        self._elapsed = 0

    def elapsed(self):
        """Get the elapsed time in milliseconds."""
        if self._start_time is not None:
            return self._elapsed + (time.perf_counter() - self._start_time) * 1000
        return self._elapsed

    def is_running(self):
        """Check if the timer is currently running."""
        return self._start_time is not None


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


###########################################
############ Global variables #############
###########################################
DROWSY = False
DISTRACTED = False
OPEN_THRESHOLD_CALIBRATED = False
CLOSED_THRESHOLD_CALIBRATED = False
OPEN_THRESHOLD_R = -1 # placeholder value
OPEN_THRESHOLD_L = -1 # placeholder value
CLOSED_THRESHOLD_R = 10000 # placeholder value
CLOSED_THRESHOLD_L = 10000 # placeholder value
EYELID_CLOSING = False
DEBUG = True # if True EAR values, PERCLOS times, nose and eyes direction are shown 

# timer instances
t_EAR = Timer()
t_cal = Timer()
t_perclos = Timer()
t_1 = Timer()
t_2 = Timer()
t_3 = Timer()
t_4 = Timer()

# variables for calibration
ear_r_calibrating_values = []
ear_l_calibrating_values = []

# variables for eyelid closure detection
ear_r_history = deque(maxlen=10)
ear_l_history = deque(maxlen=10)
time_history = deque(maxlen=10)

perclos = 0


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    # indices
    EAR_R_IDX = [33, 160, 158, 155, 153, 144]
    EAR_L_IDX = [362, 385, 387, 263, 373, 380]
    FACE_IDX = [33, 263, 1, 61, 291, 199]
    LEFT_IRIS_IDX = [473, 362, 374, 263, 386]
    RIGHT_IRIS_IDX = [468, 33, 145, 133, 159]

    # variables initialization
    right_eye_p = np.zeros((6, 2), dtype=np.float64)
    left_eye_p = np.zeros((6, 2), dtype=np.float64)
    face_2d = np.zeros((6,2), dtype=np.float64)
    face_3d = np.zeros((6,3), dtype=np.float64)
    left_eye_2d = np.zeros((5,2), dtype=np.float64)
    left_eye_3d = np.zeros((5,3), dtype=np.float64)
    right_eye_2d = np.zeros((5,2), dtype=np.float64)
    right_eye_3d = np.zeros((5,3), dtype=np.float64)

    point_RER = [] # Right Eye Right
    point_LEL = [] # Left Eye Left


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                if idx in EAR_R_IDX:
                    idx_to_add = EAR_R_IDX.index(idx)
                    right_eye_p[idx_to_add] = (lm.x, lm.y)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=1, color=(0, 0, 255), thickness=-1)

                if idx in EAR_L_IDX:
                    idx_to_add = EAR_L_IDX.index(idx)
                    left_eye_p[idx_to_add] = (lm.x, lm.y)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=1, color=(0, 0, 255), thickness=-1)

                if idx in FACE_IDX:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    idx_to_add = FACE_IDX.index(idx)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d[idx_to_add] = (x, y)
                    face_3d[idx_to_add] = (x, y, lm.z)

                if idx in LEFT_IRIS_IDX:
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    idx_to_add = LEFT_IRIS_IDX.index(idx)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    left_eye_2d[idx_to_add] = (x, y)
                    left_eye_3d[idx_to_add] = (x, y, lm.z)

                if idx in RIGHT_IRIS_IDX:
                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    idx_to_add = RIGHT_IRIS_IDX.index(idx)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    right_eye_2d[idx_to_add] = (x, y)
                    right_eye_3d[idx_to_add] = (x, y, lm.z)
                
                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)    

                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)


            # EAR calculation
            EAR_R = (np.abs(right_eye_p[1][1] - right_eye_p[5][1]) + np.abs(right_eye_p[2][1] - right_eye_p[4][1])) / (2 * np.abs(right_eye_p[0][0] - right_eye_p[3][0]))
            EAR_L = (np.abs(left_eye_p[1][1] - left_eye_p[5][1]) + np.abs(left_eye_p[2][1] - left_eye_p[4][1])) / (2 * np.abs(left_eye_p[0][0] - left_eye_p[3][0]))

            if not (OPEN_THRESHOLD_CALIBRATED and CLOSED_THRESHOLD_CALIBRATED): 
                cv2.putText(image, "Calibrating...", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # open threshold calibration
                if not OPEN_THRESHOLD_CALIBRATED:
                    cv2.putText(image, "Please keep your eyes OPEN for 3 seconds", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if not t_cal.is_running():
                        t_cal.start()
                    ear_r_calibrating_values.append(EAR_R)
                    ear_l_calibrating_values.append(EAR_L)

                    if t_cal.elapsed() > 3000:

                        # threshold as mean of the 100 biggest values
                        OPEN_THRESHOLD_L = st.mean(sorted(ear_l_calibrating_values)[-100:])
                        OPEN_THRESHOLD_R = st.mean(sorted(ear_r_calibrating_values)[-100:])
                        OPEN_THRESHOLD_CALIBRATED = True

                        # reset the calibration variables
                        ear_r_calibrating_values = []
                        ear_l_calibrating_values = []
                        t_cal.stop()
                        t_cal.reset()

                else: 
                    cv2.putText(image, "Please keep your eyes CLOSED for 3 seconds", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not t_cal.is_running():
                        t_cal.start()
                    ear_r_calibrating_values.append(EAR_R)
                    ear_l_calibrating_values.append(EAR_L)

                    if t_cal.elapsed() > 3000:
                        # threshold as mean of the 100 smallest values 
                        CLOSED_THRESHOLD_L = st.mean(sorted(ear_l_calibrating_values)[:100])
                        CLOSED_THRESHOLD_R = st.mean(sorted(ear_r_calibrating_values)[:100])
                        CLOSED_THRESHOLD_CALIBRATED = True

                        # reset the calibration variables
                        ear_r_calibrating_values = []
                        ear_l_calibrating_values = []
                        t_cal.stop()
                        t_cal.reset()

            else:
                # scale the EAR values in the range of 0 to 1
                EAR_R = (EAR_R - CLOSED_THRESHOLD_R) / (OPEN_THRESHOLD_R - CLOSED_THRESHOLD_R)
                EAR_L = (EAR_L - CLOSED_THRESHOLD_L) / (OPEN_THRESHOLD_L - CLOSED_THRESHOLD_L)
                
                # Alarm if the EAR is greater than 0.8 for more than 10 seconds, meaning the driver did not blink for a long time
                if EAR_R > 0.8 and EAR_L > 0.8:
                    t_EAR.start() if not t_EAR.is_running() else None
                else:
                    # reset when eyes are open
                    t_EAR.stop()
                    t_EAR.reset()

                if t_EAR.elapsed() > 10000:
                    DROWSY = True
                    t_EAR.stop()
                    t_EAR.reset()
                
                ################ PERCLOS CALCULATION ################
                time_history.append(time.time())
                ear_r_history.append(EAR_R)
                ear_l_history.append(EAR_L)

                # EAR derivative calculation, if positive then the eyes are closing
                if len(ear_r_history) == 10 and len(ear_l_history) == 10:
                    ear_r_derivative = (ear_r_history[-1] - ear_r_history[0]) / (time_history[-1] - time_history[0])
                    ear_l_derivative = (ear_l_history[-1] - ear_l_history[0]) / (time_history[-1] - time_history[0])

                    # eyes are closing, so start t1
                    if ear_r_derivative < -0.5 and ear_l_derivative < -0.5 and not EYELID_CLOSING:
                        EYELID_CLOSING = True
                        t_1.start() 

                # t2, t3 and t4 calculation    
                if EYELID_CLOSING:
                    mean_EAR = (EAR_R + EAR_L) / 2
                    
                    # t2 calculation
                    if mean_EAR < 0.8 and mean_EAR > 0.2:
                        t_1.stop()
                        t_2.start() 

                    # t3 and t4 calculation
                    if mean_EAR < 0.2:
                        t_2.stop()
                        t_3.start() 
                        t_4.start()

                    if mean_EAR > 0.2:
                        t_3.stop()
                    
                    if mean_EAR > 0.8:
                        t_4.stop()
                        EYELID_CLOSING = False
                        
                        # calculate perclos and reset the timers
                        if t_1.elapsed() > 0 and t_2.elapsed() > 0 and t_3.elapsed() > 0 and t_4.elapsed() > 0:
                            perclos = (t_3.elapsed() - t_2.elapsed()) / (t_4.elapsed() - t_1.elapsed()) 
                            if perclos < 0:
                                perclos = 0
                            t_1.reset()
                            t_2.reset()
                            t_3.reset()
                            t_4.reset()                

                # Display the values on the image
                if OPEN_THRESHOLD_CALIBRATED and CLOSED_THRESHOLD_CALIBRATED:
                    if not DROWSY:
                        cv2.putText(image, "Awake", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "DROWSY", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if DEBUG:
                        # Debug information
                        cv2.putText(image, f'DEBUG INFORMATION', (480, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(image, f'EAR R: {EAR_R:.2f}' , (480, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f'EAR L: {EAR_L:.2f}', (480, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f"Perclos: {perclos:.2f}", (480, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f'EYELID CLOSING: {EYELID_CLOSING}', (480, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f"t1: {t_1.elapsed():.2f}", (480, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f"t2: {t_2.elapsed():.2f}", (480, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f"t3: {t_3.elapsed():.2f}", (480, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(image, f"t4: {t_4.elapsed():.2f}", (480, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    


            ################# Detect if the driver is distracted #################
            #Camera matrix
            focal_length = img_w
            cam_matrix = np.array(
                [[focal_length, 0, img_h/2],
                [0, focal_length, img_w/2],
                [0, 0, 1]]        )

            dist_matrix = np.zeros((4,1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)

            success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
            rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)

            # Get the euler angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
            Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
            angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)
            Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)

            # Get RPY angles
            pitch = angles[0]*1800
            yaw = angles[1]*1800
            roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0] - point_LEL[0]) * 180 / np.pi)
            if roll > 180:
                roll = roll - 360

            pitch_left_eye = angles_left_eye[0]*1800
            yaw_left_eye = angles_left_eye[1]*1800

            pitch_right_eye = angles_right_eye[0]*1800
            yaw_right_eye = angles_right_eye[1]*1800

            #Display directions of nose
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1_nose = (int(nose_2d[0]), int(nose_2d[1]))
            p2_nose = (int(nose_2d[0] + yaw * 2), int(nose_2d[1] - pitch * 2))

            # Display directions of left eye
            l_eye_projection, l_eye_jacobian = cv2.projectPoints(left_eye_3d, rot_vec_left_eye, trans_vec, cam_matrix, dist_matrix)
            p1_left_eye = (int(left_pupil_2d[0]), int(left_pupil_2d[1]))
            p2_left_eye = (int(left_pupil_2d[0] + yaw_left_eye * 1.25), int(left_pupil_2d[1] - pitch_left_eye * 1.25))
            
            # Display directions of right eye
            r_eye_projection, r_eye_jacobian = cv2.projectPoints(right_eye_3d, rot_vec_right_eye, trans_vec, cam_matrix, dist_matrix)
            p1_right_eye = (int(right_pupil_2d[0]), int(right_pupil_2d[1]))
            p2_right_eye = (int(right_pupil_2d[0] + yaw_right_eye * 1.25), int(right_pupil_2d[1] - pitch_right_eye * 1.25))
            
            # check if the driver is distracted
            if np.abs(yaw) > 30 and np.abs(yaw_left_eye) > 30 and np.abs(yaw_right_eye) > 30: 
                DISTRACTED = True
            else:
                DISTRACTED = False

            end = time.time()
            totalTime = end-start

            if totalTime>0:
                fps = 1 / totalTime
            else:
                fps=0

            if DEBUG:
                cv2.line(image, p1_nose, p2_nose, (0, 255, 255), 3)
                cv2.line(image, p1_left_eye, p2_left_eye, (0, 255, 0), 3)
                cv2.line(image, p1_right_eye, p2_right_eye, (0, 255, 0), 3)
                # show yaw
                cv2.putText(image, f'Yaw: {yaw:.2f}', (0, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f'Right eye yaw: {yaw_right_eye:.2f}', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f'Left eye yaw: {yaw_left_eye:.2f}', (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if OPEN_THRESHOLD_CALIBRATED and CLOSED_THRESHOLD_CALIBRATED:
                if not DISTRACTED:
                    cv2.putText(image, "FOCUSSED", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                else:
                    cv2.putText(image, "DISTRACTED", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   

            cv2.imshow('output window', image)    


        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
