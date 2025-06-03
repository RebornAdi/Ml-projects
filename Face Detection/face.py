import face_recognition as fr
import os
import cv2
import gc

known_faces_dir = "known_faces"
unknown_faces_dir = "unknown_faces"
tolerance = 0.6  # lower value means more strict matching
frame_thickness = 3
font_thickness = 2
Model = "cnn"

print("Loading known faces")

known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):
    for filename in os.listdir(f"{known_faces_dir}/{name}"):
        image = fr.load_image_file(f"{known_faces_dir}/{name}/{filename}")
        encodings = fr.face_encodings(image)
        for encoding in encodings:
            known_faces.append(encoding)
            known_names.append(name)

print("Processing unknown faces")

# Ensure the unknown_faces_dir exists
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)
    print(f"Directory '{unknown_faces_dir}' created.")

for filename in os.listdir(unknown_faces_dir):
    try:
        print(f"Processing {filename}")
        image_path = os.path.join(unknown_faces_dir, filename)
        image = fr.load_image_file(image_path)
        locations = fr.face_locations(image, model=Model)
        encodings = fr.face_encodings(image, locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not encodings:
            print(f"No faces found in {filename}")
            continue

        for face_encoding, face_location in zip(encodings, locations):
            results = fr.compare_faces(known_faces, face_encoding, tolerance)
            match = None
            if True in results:
                match = known_names[results.index(True)]
                print(f"Match found: {match}")

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = [0, 255, 0]

                cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)

        cv2.imshow(filename, image)
        cv2.waitKey(10000)
        cv2.destroyWindow(filename)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
    
    finally:
        # Explicitly release memory
        del image, locations, encodings
        gc.collect()

cv2.destroyAllWindows()
