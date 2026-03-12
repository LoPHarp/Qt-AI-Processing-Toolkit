import os
import cv2
import pandas as pd


def create_dataset(base_dir, output_csv):
    emotion_map = {
        "anger": 0,
        "contempt": 1,
        "disgust": 2,
        "fear": 3,
        "happy": 4,
        "sadness": 5,
        "surprise": 6
    }

    data = []

    for emotion_name, label in emotion_map.items():
        folder_path = os.path.join(base_dir, emotion_name)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (48, 48))

            pixels = ' '.join(map(str, img.flatten()))
            data.append({'emotion': label, 'pixels': pixels})

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")


if __name__ == '__main__':
    create_dataset('CK+48', 'ckplus_dataset.csv')