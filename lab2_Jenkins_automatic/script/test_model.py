import pandas as pd
from sklearn.metrics import accuracy_score # коэффициент детерминации  от Scikit-learn
import joblib


def test_model():

    # Загружаем модель
    model = joblib.load('models/model.pkl')

    # Загружаем данные
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # Предсказываем на тестовых данных
    y_pred = model.predict(X_test)

    # Рассчитываем метрику
    result = accuracy_score(y_test, y_pred)
    print(f'Model accuracy :  {result}')
    

if __name__ == "__main__":
    test_model()