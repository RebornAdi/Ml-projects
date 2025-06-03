import cv2
import os
from datetime import datetime

# Emotion labels and number of samples per class
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SAMPLES_PER_CLASS = 100
DATASET_PATH = "collected_dataset"

# Create folders
for emotion in EMOTIONS:
    os.makedirs(os.path.join(DATASET_PATH, emotion), exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("Press keys to select emotion:")
for i, emo in enumerate(EMOTIONS):
    print(f"Press {i} for '{emo}'")

current_emotion = None
counters = {emo: len(os.listdir(os.path.join(DATASET_PATH, emo))) for emo in EMOTIONS}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1)

    # Set emotion from keypress
    if key in range(ord('0'), ord('0') + len(EMOTIONS)):
        current_emotion = EMOTIONS[key - ord('0')]
        print(f"🎯 Collecting for: {current_emotion}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        if current_emotion and counters[current_emotion] < SAMPLES_PER_CLASS:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
            save_path = os.path.join(DATASET_PATH, current_emotion, filename)
            cv2.imwrite(save_path, face)

            counters[current_emotion] += 1
            print(f"📸 Saved {current_emotion}: {counters[current_emotion]}/{SAMPLES_PER_CLASS}")

        # Draw rectangle on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if current_emotion:
            cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("Collecting Data - Press 'q' to Quit", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Data collection completed.")
