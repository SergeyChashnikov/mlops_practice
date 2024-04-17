import pandas as pd
from sklearn.metrics import r2_score # коэффициент детерминации  от Scikit-learn
import joblib

# Функция тестирования модели
def test_model(model_path, test_data_path):

    # Загружаем модель
    model = joblib.load(model_path)

    # Загружаем тестовый датасет
    df_test = pd.read_csv(test_data_path)

    # Разделяем данные
    X_test, y_test = df_test.drop(columns = ['Salary']), df_test['Salary'].values

    # Предсказываем на тестовых данных
    y_pred = model.predict(X_test)

    # Рассчитываем метрику
    result = r2_score(y_test, y_pred)
    
    return result

# Путь для модели
model_path = 'models/model.pkl'

# Путь для тестовых данных
test_data_path = 'data/test/data_test_preprocess.csv'

# Тестирование модели
result = test_model(model_path, test_data_path)
print()
print('Model r2-score:  ', result)
print()