import pandas as pd
from datasets import load_dataset


def download_and_prepare_dataset():
    print("Завантаження датасету емоцій...")

    dataset = load_dataset("dair-ai/emotion", "split")

    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    df = pd.concat([train_df, test_df], ignore_index=True)

    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    df['emotion_name'] = df['label'].map(emotion_map)
    df = df.rename(columns={'label': 'emotion'})

    df.to_csv("emotion_text_dataset_notclean.csv", index=False)
    print("Датасет успішно збережено у файл 'emotion_dataset.csv'!")
    print(f"Загальна кількість прикладів: {len(df)}")
    print("Перші 5 рядків:")
    print(df.head())


if __name__ == '__main__':
    download_and_prepare_dataset()