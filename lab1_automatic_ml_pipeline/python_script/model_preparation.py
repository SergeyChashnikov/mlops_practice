import pandas as pd
from sklearn.ensemble import RandomForestRegressor # Случайный Лес для Регрессии от scikit-learn
from sklearn.utils import shuffle
import joblib
import os


# Функция обучения модели и предсказания
def train_model_and_evaluate(file_path):

    # Загружакм данные
    df = pd.read_csv(file_path)

    # Перемешивам данные
    df = shuffle(df, random_state=42)

    # Разделяем данные
    X, y = df.drop(columns = ['Salary']), df['Salary'].values

    # Создаем модель Случайный Лес
    model_rf = RandomForestRegressor(n_estimators=50,
                                     max_depth=20,
                                     oob_score=True)

    # Обучаем модель
    model_rf.fit(X, y)

    return model_rf

# Создаем папку для модели
os.makedirs('models', exist_ok=True)

# Обучаем модель
model = train_model_and_evaluate(f'data/train/data_train_preprocess.csv')

# Сохраняем модель
joblib.dump(model, f'models/model.pkl')
