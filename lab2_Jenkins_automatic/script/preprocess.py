import pandas as pd # Библиотека Pandas для работы с табличными данными
from sklearn.model_selection import train_test_split # функция разбиения на тренировочную и валидационную выборку
# в исполнении scikit-learn
from sklearn.preprocessing import StandardScaler # Библиотека для стандартизации
from sklearn.preprocessing import OrdinalEncoder # Библиотека для порядковое кодирование от scikit-learn


# Функция стандартизации
def standard_data(data):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Функция порядкового кодирования
def ordinal_data(data):

    ordinaler = OrdinalEncoder()
    ordinaled_data = ordinaler.fit_transform(data)
    return ordinaled_data

def preprocess():
    data = pd.read_csv('data/salary_data.csv')
    #print(data)

    cat_columns = []
    num_columns = []

    # Создаем списки колонок по типам данных
    for column_name in data.columns:
        if (data[column_name].dtypes == object):
            cat_columns +=[column_name]
        else:
            num_columns +=[column_name]

    # Стандартизация числовых признаков
    df_num_columns = standard_data(data[num_columns])
    data[num_columns] = df_num_columns

    # Порядковое кодирование категориальных признаков
    df_cat_columns = ordinal_data(data[cat_columns])
    data[cat_columns] = df_cat_columns

    #print(data)
    X = data.drop('Education Level', axis=1)
    y = data['Education Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    for name, dataset in datasets.items():
        dataset.to_csv(f'data/{name}.csv', index=False)
    print("Data preprocess and saved in data/")


if __name__ == "__main__":
    preprocess()