import pandas as pd
import numpy as np

print("Завантаження даних...")
main_df = pd.read_csv('training.csv')
custom_df = pd.read_csv('my_custom_dataset.csv')

final_custom_df = custom_df.tail(9).copy()

final_custom_df.columns = main_df.columns

merged_df = pd.concat([main_df, final_custom_df], ignore_index=True)

merged_df.dropna(inplace=True)
print(f"Після видалення порожніх координат залишилось: {len(merged_df)} фото")

print("Перевірка цілісності пікселів (це займе пару секунд)...")

def is_valid_image(img_str):
    try:
        return len(str(img_str).split()) == 9216
    except:
        return False

merged_df = merged_df[merged_df['Image'].apply(is_valid_image)]

merged_df.to_csv('merged_dataset_clean.csv', index=False)
print(f"ГОТОВО! Збережено ідеально чистих фото для навчання: {len(merged_df)}")