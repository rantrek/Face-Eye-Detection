#Import libraries
import numpy as np
from scipy.spatial import distance as dist
from time import sleep
import cv2
import mediapipe as mp
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

