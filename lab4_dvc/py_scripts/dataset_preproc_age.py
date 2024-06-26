import pandas as pd

# Считываем данные
titanic_df = pd.read_csv('datasets/titanic_df.csv')

# Вычисление среднего значения возраста
mean_age = round(titanic_df['Age'].mean(), 0)

# Заполнение пропущенных значений средним значением
titanic_df['Age'].fillna(mean_age, inplace=True)

titanic_df.to_csv(f'datasets/titanic_df.csv', index=False)