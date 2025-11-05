import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os

with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file) or pd.read_csv(attendance_file).columns.tolist() != ["Name", "Date", "Time"]:
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time\n")

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        new_row = pd.DataFrame([[name, today, current_time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"[✓] Marked attendance for {name} at {current_time}")
    else:
        print(f"[INFO] {name} already marked for {today}")

cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        print("[DEBUG] Matches:", matches)
        print("[DEBUG] Distances:", face_distances)

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)

        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Attendance session ended.")
