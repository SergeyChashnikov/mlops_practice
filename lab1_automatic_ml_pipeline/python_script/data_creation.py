import pandas as pd # Библиотека Pandas для работы с табличными данными
import os # Библиотека Os для работы с операционной системой
from sklearn.model_selection import train_test_split # функция разбиения на тренировочную и валидационную выборку
# в исполнении scikit-learn


# Создание отдельной папки для данных
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

LINK = 'https://raw.githubusercontent.com/SergeyChashnikov/URFUML2023_STUDIES/main/MOMO/1_semestr/Salary_Data/Salary_Data_clear.csv'

# Функция загрузки данных
def load_data(link):
    if link == None :
        print(' adress error ')
    else:
        data = pd.read_csv(link, delimiter = ',')
    return data

# Загружаем данные
Data = load_data(LINK)

# Разбиваем на тренировочную и тестовую
df_train, df_test = train_test_split(Data, test_size=0.3, random_state=42)

# Сохраняем данные
df_train.to_csv(f'data/train/data_train.csv', index=False)
df_test.to_csv(f'data/test/data_test.csv', index=False)



