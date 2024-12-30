import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import win32gui
import win32con
import win32api

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        
        self.shrimp_count = 0
        
        self.status_images = {}
        required_images = ['init.png', 'good.png', 'bad.png']
        
        for img_name in required_images:
            try:
                img = cv2.imread(img_name)
                if img is None:
                    raise FileNotFoundError
                self.status_images[img_name] = cv2.resize(img, (150, 150))
                
                img_with_counter = np.zeros((200, 150, 3), dtype=np.uint8)
                img_with_counter[0:150, :] = self.status_images[img_name]
                self.status_images[img_name] = img_with_counter
                
            except FileNotFoundError:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Error", f"{img_name} not found in current directory!")
                exit(1)
        
        self.screen = tk.Tk()
        self.screen_width = self.screen.winfo_screenwidth()
        self.screen_height = self.screen.winfo_screenheight()
        self.screen.destroy()
        
        cv2.namedWindow('Posture Status', cv2.WINDOW_NORMAL)
        
        hwnd = win32gui.FindWindow(None, 'Posture Status')
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | win32con.WS_SYSMENU)
        win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
        
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, self.screen_width - 170, 10, 150, 200, win32con.SWP_NOSIZE)
        
        self._update_status_image('init.png')
        
    def _update_status_image(self, image_name):
        """Update the status window with image and counter"""
        image = self.status_images[image_name].copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Shrimp Count: {self.shrimp_count}"
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        x = (150 - text_width) // 2
        y = 180
        
        cv2.rectangle(image, (0, 150), (150, 200), (0, 0, 0), -1)
        
        cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imshow('Posture Status', image)
        
    def check_posture(self, landmarks):
        """
        Check if posture is correct based on multiple alignment factors
        Returns True if posture is good, False otherwise
        """
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y) / shoulder_width
        
        nose_forward = (nose.z - (left_shoulder.z + right_shoulder.z) / 2) / shoulder_width
        
        head_height = ((left_ear.y + right_ear.y) / 2 - (left_shoulder.y + right_shoulder.y) / 2) / shoulder_width
        
        ear_distance = abs(left_ear.x - right_ear.x)
        head_rotation = abs(1 - (ear_distance / shoulder_width))
        
        SHOULDER_THRESHOLD = 0.3
        HEAD_FORWARD_THRESHOLD = 2.0
        HEAD_HEIGHT_THRESHOLD = 0.9
        HEAD_ROTATION_THRESHOLD = 0.7
        
        shoulders_level = shoulder_slope < SHOULDER_THRESHOLD
        head_not_forward = abs(nose_forward) < HEAD_FORWARD_THRESHOLD
        head_not_low = abs(head_height) < HEAD_HEIGHT_THRESHOLD
        normal_rotation = head_rotation < HEAD_ROTATION_THRESHOLD
        
        if normal_rotation:
            return shoulders_level and head_not_forward and head_not_low
        else:
            return shoulders_level and head_not_low
    
    def run(self):
        bad_posture_counter = 0
        current_status = 'init'
        hwnd = win32gui.FindWindow(None, 'Posture Status')
        was_bad_posture = False
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, self.screen_width - 170, 10, 150, 200, win32con.SWP_NOSIZE)
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                good_posture = self.check_posture(results.pose_landmarks.landmark)
                
                if not good_posture:
                    bad_posture_counter = min(12, bad_posture_counter + 1)
                    was_bad_posture = bad_posture_counter > 10
                else:
                    if was_bad_posture:
                        self.shrimp_count += 1
                    bad_posture_counter = 0
                    was_bad_posture = False
                
                if bad_posture_counter > 10 and current_status != 'bad':
                    current_status = 'bad'
                    self._update_status_image('bad.png')
                    print("🚫 Bad posture detected.")
                elif bad_posture_counter == 0 and current_status != 'good' and current_status != 'init':
                    current_status = 'good'
                    self._update_status_image('good.png')
                    print("✅ Posture corrected.")
            
            debug_frame = frame.copy()
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    debug_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
            cv2.imshow('Debug View', debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PostureDetector()
    detector.run()