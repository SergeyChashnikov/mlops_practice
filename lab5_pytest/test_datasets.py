
# Ваш код с тестами здесь
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error

def test_dataset(df='df'):

  df = pd.read_csv(df)

  X_test = df.drop(columns='Target')
  y_test= df['Target']

  # Предсказание на тестовом наборе
  y_pred = model.predict(X_test)

  # Вычисление MSE
  mse = mean_squared_error(y_test, y_pred)
  print("Mean Squared Error:", mse)

  assert mse < 0.22, 'MSE превышает 0.22'

# Загрузка модели из файла
model = load('linear_regression_model.joblib')

test_dataset(df='df')
test_dataset(df='df1')
test_dataset(df='df2')
test_dataset(df='df3')
