
import os
import cv2
import face_recognition
import pickle
import numpy as np

known_encodings = []
known_names = []

dataset_path = "E:/face_rec/dataset"

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        image = cv2.imread(img_path)

        if image is None or not isinstance(image, np.ndarray):
            print(f"[!] Invalid image at {img_path}, skipped.")
            continue

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"[!] cvtColor error at {img_path}, skipped.")
            continue

        try:
            encodings = face_recognition.face_encodings(rgb)
        except Exception as e:
            print(f"[!] face_recognition failed at {img_path}: {e}")
            continue

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"[!] No face found in {img_path}, skipped.")

print(f"[INFO] Successfully encoded {len(known_names)} face images.")

with open("encodings.pickle", "wb") as f:
    pickle.dump((known_encodings, known_names), f)
