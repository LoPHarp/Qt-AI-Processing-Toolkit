import pandas as pd
import re


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset(input_file, output_file):
    print(f"Читання датасету з {input_file}...")
    df = pd.read_csv(input_file)

    print("Початок очищення тексту...")
    df['clean_text'] = df['text'].apply(clean_text)

    df = df[df['clean_text'].str.len() > 0]

    df.to_csv(output_file, index=False)
    print(f"Очищений датасет збережено у {output_file}!")

    print("\nПриклад того, як змінився текст:")
    for i in range(3):
        print(f"ОРИГІНАЛ: {df['text'].iloc[i]}")
        print(f"ОЧИЩЕНО:  {df['clean_text'].iloc[i]}\n")


if __name__ == '__main__':
    preprocess_dataset('emotion_text_dataset_notclean.csv', 'emotion_text_dataset.csv')