from catboost.datasets import titanic
import pandas as pd
import os

# Получаем датасет
titanic_train, titanic_test = titanic()
titanic_df = pd.concat([titanic_train, titanic_test], ignore_index=True)

# Создание отдельной папки для данных
os.makedirs('datasets', exist_ok=True)

# Сохраняем данные
titanic_df.to_csv(f'datasets/titanic_df.csv', index=False)
