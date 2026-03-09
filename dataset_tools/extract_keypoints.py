import cv2
import os
import pandas as pd
import numpy as np
import onnxruntime as ort

emotion_map = {
    'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3,
    'happy': 4, 'sadness': 5, 'surprise': 6
}

model_path = 'model_Accuracy_08.onnx'
cascade_path = 'haarcascade_frontalface_default.xml'
dataset_path = 'CK+48'

face_cascade = cv2.CascadeClassifier(cascade_path)
ort_session = ort.InferenceSession(model_path)

data = []

for emotion_name, emotion_label in emotion_map.items():
    folder_path = os.path.join(dataset_path, emotion_name)

    if not os.path.exists(folder_path): continue

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None: continue

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = img[y:y + h, x:x + w]
        else:
            face_crop = img

        face_resized = cv2.resize(face_crop, (96, 96))

        blob = cv2.dnn.blobFromImage(face_resized, 1.0 / 255.0, (96, 96), (0, 0, 0), swapRB=False, crop=False)

        input_name = ort_session.get_inputs()[0].name
        keypoints = ort_session.run(None, {input_name: blob})[0]

        row = [emotion_label] + keypoints[0].tolist()
        data.append(row)

    print(f"Оброблено емоцію: {emotion_name}")

columns = ['emotion']
for i in range(1, 16):
    columns.extend([f'x{i}', f'y{i}'])

df = pd.DataFrame(data, columns=columns)
df.to_csv('emotion_keypoints.csv', index=False)
print("Готово! Дані збережено у emotion_keypoints.csv")