from catboost.datasets import titanic
import os

titanic_train, titanic_test = titanic()
print(titanic_train, titanic_test)

# Создание отдельной папки для данных
os.makedirs('datasets', exist_ok=True)

# Сохраняем данные
titanic_train.to_csv(f'datasets/titanic_train.csv', index=False)
titanic_test.to_csv(f'datasets/titanic_test.csv', index=False)