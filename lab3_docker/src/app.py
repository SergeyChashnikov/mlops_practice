import streamlit as st
import pandas as pd
import json
import joblib

# Функция для загрузки модели
def load_model():
    model = joblib.load('app/data/model.pkl')
    return model

model = load_model()

st.title('Education level Prediction')
st.write('Upload a JSON file for prediction')

# Загрузка файла пользователя
upload_file = st.file_uploader('Chooser a JSON file', type=['json'])
if upload_file is not None:
    data = json.load(upload_file)
    df = pd.DataFrame([data])

    try:
        # Предсказание
        prediction = model.predict(df[['Age', 'Gender', 'Job Title', 'Years of Experience', 'Salary']])
        st.write('Prediction Education level: ', prediction[0])
    except Exception as e:
        st.error(f"Error of prediction: {e}")
    finally:
        upload_file.seek(0) # Возвращаемся к началу файла