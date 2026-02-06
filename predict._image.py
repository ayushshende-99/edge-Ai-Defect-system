import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/wafer_model.h5")
class_names = ['Center', 'Edge_Ring', 'Scratch']

img = cv2.imread("test_image.jpg")

if img is None:
    print("❌ Image not found. Check file path.")
    exit()

img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Prediction:", class_names[np.argmax(prediction)])