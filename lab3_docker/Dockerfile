FROM python:3.8

# Установка зависимостей
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# Копирование файлов проекта
COPY . /app
WORKDIR /app

# Обучение модели
RUN python /app/src/train_model.py

# Запуск Streamlit
CMD ["streamlit", "run", "src/app.py" ]

