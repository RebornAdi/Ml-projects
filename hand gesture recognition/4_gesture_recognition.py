import cv2
import mediapipe as mp
import joblib
import numpy as np
from config import (
    GESTURE_LABELS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS
)

def main():
    # Load trained model
    model = joblib.load('models/gesture_model.pkl')
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image)
        
        # Draw hand landmarks and recognize gesture
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks for classification
                data = []
                for landmark in hand_landmarks.landmark:
                    data.extend([landmark.x, landmark.y, landmark.z])
                
                # Predict gesture
                gesture_idx = model.predict([data])[0]
                gesture_label = GESTURE_LABELS.get(gesture_idx, "Unknown")
                
                # Display prediction
                cv2.putText(image, f'Gesture: {gesture_label}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Print to console
                print(f'Detected Gesture: {gesture_label}')
        
        # Display the image
        cv2.imshow('Hand Gesture Recognition', image)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()