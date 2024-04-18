import pandas as pd # Библиотека Pandas для работы с табличными данными
import os # Библиотека Os для работы с операционной системой


# Создание отдельной папки для данных
os.makedirs('data', exist_ok=True)

# Ссылка для загрузки данных
LINK = 'https://raw.githubusercontent.com/SergeyChashnikov/URFUML2023_STUDIES/main/MOMO/1_semestr/Salary_Data/Salary_Data_clear.csv'

# Функция загрузки данных
def load_data(link):
    if link == None :
        print(' adress error ')
    else:
        data = pd.read_csv(link, delimiter = ',')
        data.to_csv('data/salary_data.csv', index=False)
        print("Data loaded and saved in data/salary_data.csv")
    
if __name__ == "__main__":
    load_data(LINK)