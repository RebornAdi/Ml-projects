import cv2

def detect_faces(gray_frame, face_cascade):
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return faces
