import os
import numpy as np
import cv2
import mediapipe as mp
from config import (
    NUM_GESTURES, 
    SAMPLES_PER_GESTURE,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS
)

def initialize_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

def main():
    # Create directory for gesture data
    DATA_DIR = './data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    hands = initialize_hands()
    cap = cv2.VideoCapture(0)

    for gesture_idx in range(NUM_GESTURES):
        # Create directory for each gesture
        gesture_dir = os.path.join(DATA_DIR, str(gesture_idx))
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
        
        print(f'Collecting data for gesture #{gesture_idx}')
        print('Press "s" to start collecting...')
        
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Gesture #{gesture_idx}. Press "s" to start', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('s'):
                break
        
        # Collect samples
        counter = 0
        while counter < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Save landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                data = []
                for landmark in hand_landmarks.landmark:
                    data.extend([landmark.x, landmark.y, landmark.z])
                
                np.save(os.path.join(gesture_dir, f'{counter}.npy'), np.array(data))
                counter += 1
                print(f'Collected {counter}/{SAMPLES_PER_GESTURE}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()