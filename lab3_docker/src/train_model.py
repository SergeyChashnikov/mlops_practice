import pandas as pd # Библиотека Pandas для работы с табличными данными
import os # Библиотека Os для работы с операционной системой
from sklearn.model_selection import train_test_split # функция разбиения на тренировочную и валидационную выборку
# в исполнении scikit-learn
from sklearn.preprocessing import StandardScaler # Библиотека для стандартизации
from sklearn.preprocessing import OrdinalEncoder # Библиотека для порядковое кодирование от scikit-learn
from sklearn.ensemble import RandomForestClassifier # Случайный лес для классификации
import joblib # Для сохранения модели


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


# Создание отдельной папки для модели
os.makedirs('data', exist_ok=True)

# Ссылка для загрузки данных
LINK = 'https://raw.githubusercontent.com/SergeyChashnikov/URFUML2023_STUDIES/main/MOMO/1_semestr/Salary_Data/Salary_Data_clear.csv'

# Загрузка данных
data = pd.read_csv(LINK, delimiter = ',')

# Предобработка данных
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
#print()
X = data.drop('Education Level', axis=1)
y = data['Education Level']

# Разбиение на тестировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
datasets = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
    }

# Обучение модели
model_rf = RandomForestClassifier(n_estimators=150,
                                     max_depth=10,
                                     oob_score=True)
    
model_rf.fit(X_train, y_train.values.ravel())
#print(model_rf.oob_score_)
#print()

# Сохранение модели
joblib.dump(model_rf, f'app/data/model.pkl')
