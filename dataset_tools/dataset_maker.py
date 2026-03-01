import cv2
import os
import numpy as np
import pandas as pd
import glob

point_names = [
    "1. Left Eye Center", "2. Right Eye Center",
    "3. Left Eye Inner", "4. Left Eye Outer",
    "5. Right Eye Inner", "6. Right Eye Outer",
    "7. L Eyebrow Inner", "8. L Eyebrow Outer",
    "9. R Eyebrow Inner", "10. R Eyebrow Outer",
    "11. Nose Tip",
    "12. Mouth Left Corner", "13. Mouth Right Corner",
    "14. Mouth Top Lip", "15. Mouth Bottom Lip"
]

points = []
img_display = None


def mouse_click(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 15:
            points.append((x / 5.0, y / 5.0))

            cv2.circle(img_display, (x, y), 4, (0, 255, 0), -1)
            cv2.rectangle(img_display, (0, 0), (480, 40), (0, 0, 0), -1)

            if len(points) < 15:
                text = f"Click: {point_names[len(points)]}"
                color = (255, 255, 255)
            else:
                text = "Done! Press SPACE"
                color = (0, 255, 0)

            cv2.putText(img_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Annotator", img_display)


def main():
    global points, img_display

    csv_filename = "my_custom_dataset.csv"

    if not os.path.exists("my_faces"):
        os.makedirs("my_faces")
        print("Папка 'my_faces' створена. Покладіть туди фотографії.")
        return

    image_files = glob.glob("my_faces/*.*")
    if len(image_files) == 0:
        print("Папка 'my_faces' порожня!")
        return

    print("--- Попередня перевірка файлів ---")
    valid_image_files = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ПОМИЛКА] Не вдалося відкрити: {img_path} (Перевірте формат або приберіть кирилицю)")
        else:
            valid_image_files.append(img_path)
    print("----------------------------------\n")

    if len(valid_image_files) == 0:
        print("Немає жодного робочого фото для розмітки. Вихід.")
        return

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_click)

    for img_path in valid_image_files:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            print(f"Обличчя не знайдено на фото {img_path}. Пропускаємо...")
            continue

        x, y, w, h = faces[0]
        face_crop = gray[y:y + h, x:x + w]
        face_96 = cv2.resize(face_crop, (96, 96))

        img_display = cv2.resize(face_96, (480, 480))
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

        points = []
        cv2.rectangle(img_display, (0, 0), (480, 40), (0, 0, 0), -1)
        cv2.putText(img_display, f"Click: {point_names[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)
        cv2.imshow("Annotator", img_display)

        print(f"Розмічаємо: {img_path}")

        skip_image = False
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and len(points) == 15:
                break
            elif key == 27:
                print("Роботу перервано користувачем.")
                skip_image = True
                break

        if skip_image:
            break

        row_data = {}
        for i, name in enumerate(point_names):
            clean_name = name.split(". ")[1].replace(" ", "_").lower()
            row_data[f"{clean_name}_x"] = points[i][0]
            row_data[f"{clean_name}_y"] = points[i][1]

        pixel_str = " ".join(map(str, face_96.flatten()))
        row_data["Image"] = pixel_str

        df_row = pd.DataFrame([row_data])

        # Миттєве збереження після кожної картинки
        file_exists = os.path.exists(csv_filename)
        df_row.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

        print(f"Фото {img_path} успішно збережено у CSV!")

    print("Всі знайдені фото оброблені.")


if __name__ == "__main__":
    main()