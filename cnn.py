
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
print("Starting script...")

model = load_model('eye_gaze_model (1).keras')
print("Model loaded successfully.")


# Class names in the same order as used during training
class_names = ['Left', 'Right']  # Add more if your model has more classes

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indexes
LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153]
RIGHT_EYE_LANDMARKS = [263, 362, 387, 386, 385, 384, 398, 382, 381, 380, 374]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    print("Webcam is running...")


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for eye_landmarks, color, eye_label in [
                (LEFT_EYE_LANDMARKS, (0, 255, 0), "Left"),
                (RIGHT_EYE_LANDMARKS, (0, 0, 255), "Right")
            ]:
                eye_points = []
                for idx in eye_landmarks:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    eye_points.append((x, y))

                x_min = max(min(p[0] for p in eye_points), 0)
                y_min = max(min(p[1] for p in eye_points), 0)
                x_max = min(max(p[0] for p in eye_points), w)
                y_max = min(max(p[1] for p in eye_points), h)

                # Draw rectangle around eye
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Extract and preprocess eye region
                eye_img = frame[y_min:y_max, x_min:x_max]

                try:
                    eye_resized = cv2.resize(eye_img, (256, 256))
                    eye_normalized = eye_resized / 255.0
                    eye_input = np.reshape(eye_normalized, (1, 256, 256, 3))

                    prediction = model.predict(eye_input, verbose=0)
                    class_index = np.argmax(prediction[0])
                    label = class_names[class_index]
                    confidence = prediction[0][class_index]

                    # Display prediction
                    print(f"{eye_label} Eye Prediction: {label} ({confidence:.2f})")

                    cv2.putText(frame, f"{eye_label}: {label}",
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                except Exception as e:
                    print(f"Error processing {eye_label} eye: {e}")

    # Show the frame
    cv2.imshow("Eye Gaze Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
