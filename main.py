import os
import sys
import cv2
import numpy as np
import random
import mediapipe as mp
import pygame
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from absl import logging

# Suppress TensorFlow and Mediapipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity(logging.ERROR)

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize pygame for sound effects
pygame.mixer.init()
win_sound = "win.wav"
lose_sound = "lose.wav"
tie_sound = "tie.wav"

class RockPaperScissorsGame(QWidget):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.setWindowTitle("Rock-Paper-Scissors 🎮")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background-color: #222831; color: white; font-family: Arial;")

        # Video Label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #00adb5; border-radius: 10px;")

        # Labels
        self.result_label = QLabel("Make a move! ✋", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 18))
        self.result_label.setStyleSheet("color: yellow; padding: 10px;")

        self.ai_label = QLabel("AI Move: ❓", self)
        self.ai_label.setAlignment(Qt.AlignCenter)
        self.ai_label.setFont(QFont("Arial", 16))
        self.ai_label.setStyleSheet("color: cyan; padding: 10px;")

        self.score_label = QLabel("Wins: 0 | Losses: 0 | Ties: 0", self)
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 16))
        self.score_label.setStyleSheet("color: lime; padding: 10px;")

        # Layout
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.result_label)
        info_layout.addWidget(self.ai_label)
        info_layout.addWidget(self.score_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(info_layout)
        self.setLayout(main_layout)

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # State Variables
        self.previous_move = "None"
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.timer.start(30)  # Start camera feed immediately

    def update_frame(self):
        """Updates the camera frame and processes gestures."""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        user_move = self.detect_gesture(frame)

        if user_move != "None" and user_move != self.previous_move:
            self.previous_move = user_move  # Store last move
            ai_move = random.choice(["Rock", "Paper", "Scissors"])
            result_text = self.get_winner(user_move, ai_move)

            # Update Score
            if "You Win" in result_text:
                self.wins += 1
                pygame.mixer.Sound(win_sound).play()
            elif "AI Wins" in result_text:
                self.losses += 1  # Correctly updating losses
                pygame.mixer.Sound(lose_sound).play()
            else:
                self.ties += 1
                pygame.mixer.Sound(tie_sound).play()

            # Update Labels
            self.result_label.setText(f"Your Move: {user_move} 🎯")
            self.ai_label.setText(f"AI Move: {ai_move} 🤖 | {result_text}")
            self.score_label.setText(f"Wins: {self.wins} | Losses: {self.losses} | Ties: {self.ties}")

        # Convert frame to PyQt format
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def detect_gesture(self, frame):
        """Detects hand gestures using Mediapipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get finger states
                fingers = []
                tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

                # Thumb
                if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                    fingers.append(1)  # Open
                else:
                    fingers.append(0)  # Closed

                # Other four fingers
                for i in range(1, 5):
                    if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                        fingers.append(1)  # Open
                    else:
                        fingers.append(0)  # Closed

                # Recognize Gesture
                if fingers == [0, 0, 0, 0, 0]:  # All fingers closed
                    return "Rock"
                elif fingers == [1, 1, 1, 1, 1]:  # All fingers open
                    return "Paper"
                elif fingers == [0, 1, 1, 0, 0]:  # Only index and middle open
                    return "Scissors"

        return "None"

    def get_winner(self, user, ai):
        """Determines the winner based on user and AI moves."""
        if user == ai:
            return "It's a Tie! 🎲"
        elif (user == "Rock" and ai == "Scissors") or (user == "Paper" and ai == "Rock") or (user == "Scissors" and ai == "Paper"):
            return "You Win! 🏆"
        else:
            return "AI Wins! 🤖"

    def closeEvent(self, event):
        """Closes the camera when the window is closed."""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RockPaperScissorsGame()
    window.show()
    sys.exit(app.exec_())
