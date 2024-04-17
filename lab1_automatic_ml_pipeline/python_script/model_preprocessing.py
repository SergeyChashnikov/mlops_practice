import pandas as pd # Библиотека Pandas для работы с табличными данными
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

# Функция предобработки данных
def preprocess_data(path):
    # Считываем файл
    data = pd.read_csv(path)

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

    return data

# Обрабатываем и сохраняем тренировочные данные
data_train_preprocess = preprocess_data("data/train/data_train.csv")
data_train_preprocess.to_csv(f'data/train/data_train_preprocess.csv', index=False)

# Обрабатываем и сохраняем тестовые данные
data_test_preprocess = preprocess_data("data/test/data_test.csv")
data_test_preprocess.to_csv(f'data/test/data_test_preprocess.csv', index=False)
