import cv2
from attendance import recognize_faces

cap = cv2.VideoCapture(0)
print("[INFO] Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    locations, persons = recognize_faces(frame)

    for (top, right, bottom, left), (pid, name) in zip(locations, persons):
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        label = f"{pid} - {name}"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Scan Logger", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Session ended")
