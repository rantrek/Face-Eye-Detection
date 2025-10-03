#Import libraries
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import imutils
from time import sleep
import cv2
import mediapipe as mp
from threading import Thread
import playsound

#Initialize the face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

#Functions

def calculateEAR(eye):

	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def soundAlarm(path):
	# play an alarm sound
	playsound.playsound(path)
     
def createFaceMesh(img, landmarks, mesh):
    """This function detects and draws the facial landmarks on the image.
       Input: image/frame, face landmarks, mesh (mp.solutions.face_mesh)
    """

    #Initialize the drawing utitlies and styles to detect and draw facial landmarks on the image
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=landmarks,
        connections=mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    
def detectBlinks(img,landmarks):
    
    #Eye landmark indices - EAR
    LEFT_EYE_ind = [362, 380, 374, 263, 386, 385]
    RIGHT_EYE_ind = [33, 159, 158, 133, 153, 145]

    # Get the coordinates of the left and right eye landmarks
    left_eye_coord = [landmarks.landmark[i] for i in LEFT_EYE_ind]
    right_eye_coord = [landmarks.landmark[i] for i in RIGHT_EYE_ind]

    # Convert the normalized landmarks to pixel coordinates
    h = img.shape[0]
    w = img.shape[1]
    LEFT_EYE = [(int(landmark.x * w), int(landmark.y * h)) for landmark in left_eye_coord]
    RIGHT_EYE = [(int(landmark.x * w), int(landmark.y * h)) for landmark in right_eye_coord]

    leftEAR = calculateEAR(LEFT_EYE)
    rightEAR = calculateEAR(RIGHT_EYE)
    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0
    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    leftEyeHull = cv2.convexHull(np.array(LEFT_EYE))
    rightEyeHull = cv2.convexHull(np.array(RIGHT_EYE))
    cv2.drawContours(img, [leftEyeHull], -1, (255,0, 0), 1)
    cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
    
    return ear

def DrowsinessDetection(cap,model,mesh, alarm):
    """This function generates a face mesh and displays it in real-time.
       Input: webcam capture, face mesh model, mesh (mp.solutions.face_mesh)
       Returns: list of all facial landmarks for every frame
    """
    #Define two constants, one for the eye aspect ratio to indicate
    #blink and then a second constant for the number of consecutive
    #frames the eye must be below the threshold
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3

    #Initialize the frame counters 
    COUNTER = 0
    ALARM_ON = False

    with model  as face_mesh:
        eye_landmark_list =[]
        while True:
            frame = cap.read()
            
            frame = imutils.resize(frame, width=450)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for eye_landmark in results.multi_face_landmarks:
                    createFaceMesh(frame,eye_landmark,mesh)
                    ear = detectBlinks(frame,eye_landmark)
                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    # if the eyes were closed for a sufficient number of
			        # then sound the alarm
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            # if the alarm is not on, turn it on
                            if not ALARM_ON:
                                ALARM_ON = True
                                # check to see if an alarm file was supplied,
                                # and if so, start a thread to have the alarm
                                # sound played in the background
                                if alarm != "":
                                    t = Thread(target=soundAlarm,
                                        args= (alarm,))
                                    t.deamon = True
                                    t.start()
                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        COUNTER = 0
                        ALARM_ON = False
                
                    # the computed eye aspect ratio for the frame
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    eye_landmark_list.append(eye_landmark.landmark)

            cv2.imshow('MediaPipe Face Mesh', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()
        cap.stop()

        return eye_landmark_list

#Main
#set alarm path
alarm_path = "alarm.wav"
video_streaming = VideoStream(src=0).start()
#video_streaming = VideoStream(usePiCamera=True).start()
sleep(1.0)
Landmarks_list = DrowsinessDetection(video_streaming, face_mesh_model,mp_face_mesh, alarm_path)
print("Number of frames processed: ",len(Landmarks_list))

