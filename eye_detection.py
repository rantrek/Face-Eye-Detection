#Import libraries
from picamera2 import Picamera2
import imutils
from time import sleep
import cv2
import mediapipe as mp

#Initialize the face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

#Functions
def createFaceMesh(img, landmarks, mesh):
    """This function detects and draws the facial landmarks on the image.
       Input: image/frame, face landmarks, mesh (mp.solutions.face_mesh)
    """

    #Initialize the drawing utitlies and styles to detect and draw facial landmarks on the image
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    #mp_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) 

    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=landmarks,
        connections=mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    
def drawEyeLandMarks(img,landmarks):

      # Eye landmark indices (left and right eye)
    left_eye_ind = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] 
    right_eye_ind = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  

    # Draw circles on the eye landmarks
    for i in left_eye_ind + right_eye_ind:
        x = int(landmarks.landmark[i].x * img.shape[1])
        y = int(landmarks.landmark[i].y * img.shape[0])
        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)

def detectEyeLandmarks(cap,model,mesh):
    """This function generates a face mesh and displays it in real-time.
       Input: webcam capture, face mesh model, mesh (mp.solutions.face_mesh)
       Returns: list of all facial landmarks for every frame
    """
    with model  as face_mesh:
        eye_landmark_list =[]
        while True:
            #frame = cap.read() #uncomment for accessing webcam
            frame = cap.capture_array() #use for raspberry pi camera
            frame = imutils.resize(frame, width=450)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for eye_landmark in results.multi_face_landmarks:
                    createFaceMesh(frame,eye_landmark,mesh)
                    drawEyeLandMarks(frame,eye_landmark)
                    eye_landmark_list.append(eye_landmark.landmark)

            cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()
    

        return eye_landmark_list

#Main
#video_streaming = cv2.VideoCapture(0) #uncomment when using webcam
video_streaming = Picamera2() #use only for raspberry pi camera
video_streaming.start() #only for raspberry pi camera
sleep(1.0)
Landmarks_list = detectEyeLandmarks(video_streaming, face_mesh_model,mp_face_mesh)
video_streaming.stop()
print("Number of frames processed: ",len(Landmarks_list))
