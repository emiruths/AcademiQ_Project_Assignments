import cv2
import mediapipe as mp
import numpy as np

# Mediapip yüz ve göz modeli
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Göz koordinatları
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def detect_gaze(eye_landmarks):
    left, right = eye_landmarks
    gaze_ratio = abs(left[0] - right[0])
    center_x = (left[0] + right[0]) / 2

    if center_x < 0.4:
        return "LEFT"
    elif center_x > 0.6:
        return "RIGHT"
    else:
        return "CENTER"
# Kamera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

# Görüntüyü işleme
    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = image.shape

            left_eye = []
            right_eye = []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                left_eye.append((x / iw, y / ih))
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                right_eye.append((x / iw, y / ih))
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            gaze_direction = detect_gaze([left_eye[0], left_eye[1]])
            cv2.putText(image, f"Gaze: {gaze_direction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Eye Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
