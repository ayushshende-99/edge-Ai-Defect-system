import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/wafer_model.h5")

# MUST MATCH training folder order
class_names = ['Center', 'Edge_Ring', 'Scratch']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    label = class_names[np.argmax(prediction)]

    cv2.putText(
        frame,
        f"Prediction: {label}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Edge AI Defect Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
prediction = model.predict(img, verbose=0)
print(prediction)

label_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("Prediction:", class_names[label_index])
print("Confidence:", confidence)