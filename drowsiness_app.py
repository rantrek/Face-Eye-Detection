# Import libraries
import tkinter as tk
from PIL import Image, ImageTk
import imutils
from imutils.video import VideoStream
#from picamera2 import Picamera2
from threading import Thread
from drowsiness_detect_2 import *

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Blink & Drowsiness Detection")
        self.root.geometry("800x600")
        
        # Detection variables
        self.counter = 0
        self.alarm_on = False
        self.blink_count = 0
        self.ear_value = 0.0
        self.drowsiness_detected = False
        
        # Video stream
        self.cap = VideoStream(src=0).start()
        #self.cap = Picamera2()
        #self.cap.start()
        sleep(1.0)
        
        # Alarm path
        self.alarm_path = "alarm.wav"
        
        # Create UI
        self.create_widgets()
        
        # Start video update
        self.update_frame()
        
    def create_widgets(self):
        # Video display frame
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        # Info frame
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(self.info_frame, text="Detection Status", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # EAR value
        self.ear_label = tk.Label(self.info_frame, text="EAR: 0.00", font=("Arial", 14))
        self.ear_label.pack(pady=10)
        
        # Blink count
        self.blink_label = tk.Label(self.info_frame, text="Blinks: 0", font=("Arial", 14))
        self.blink_label.pack(pady=10)
        
        # Drowsiness status
        self.status_label = tk.Label(self.info_frame, text="Status: Awake", font=("Arial", 14, "bold"), fg="green")
        self.status_label.pack(pady=10)
        
        # Alert label
        self.alert_label = tk.Label(self.info_frame, text="", font=("Arial", 16, "bold"), fg="red")
        self.alert_label.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(self.info_frame, text="Press 'Q' to quit", font=("Arial", 10))
        instructions.pack(side=tk.BOTTOM, pady=10)
        
    def update_frame(self):
        frame = self.cap.read()
        
        if frame is not None:
            frame = imutils.resize(frame, width=450)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            results = face_mesh_model.process(frame_rgb)
            
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    createFaceMesh(frame_bgr, face_landmarks, mp_face_mesh)
                    ear = detectBlinks(frame_bgr, face_landmarks)
                    
                    self.ear_value = ear
                    
                    if ear < EYE_AR_THRESH:
                        self.counter += 1
                        if self.counter >= EYE_AR_CONSEC_FRAMES:
                            if not self.alarm_on:
                                self.alarm_on = True
                                self.drowsiness_detected = True
                                if self.alarm_path:
                                    t = Thread(target=soundAlarm, args=(self.alarm_path,))
                                    t.daemon = True
                                    t.start()
                            
                            #cv2.putText(frame_bgr, "DROWSINESS ALERT!", (10, 30),
                                     # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if self.counter >= EYE_AR_CONSEC_FRAMES:
                            self.blink_count += 1
                        self.counter = 0
                        self.alarm_on = False
                        self.drowsiness_detected = False
                    
                    #cv2.putText(frame_bgr, "EAR: {:.2f}".format(ear), (300, 30),
                     #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert to PIL Image for Tkinter
            frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb_display)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)
            
            # Update info labels
            self.ear_label.config(text="EAR: {:.2f}".format(self.ear_value))
            self.blink_label.config(text="Blinks: {}".format(self.blink_count))
            
            if self.drowsiness_detected:
                self.status_label.config(text="Status: Drowsy", fg="red")
                self.alert_label.config(text="DROWSINESS DETECTED!")
            else:
                self.status_label.config(text="Status: Awake", fg="green")
                self.alert_label.config(text="")
        
        # Check for quit key
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('Q', lambda e: self.quit_app())
        
        # Schedule next update
        self.root.after(10, self.update_frame)
    
    def quit_app(self):
        self.cap.stop()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()
