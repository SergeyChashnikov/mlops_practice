#!/bin/bash

# Функция создания виртуальной среды
create_venv() {
    local env_name=${1:-".venv"}
    python3 -m venv "$env_name"
    echo "The virtual environment '$env_name' has been created."
}

# Функция активации виртуальной среды
activate_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create."
        return 1
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        source "./$env_name/bin/activate"
        echo "Virtual environment '$env_name' is activated."
    else
        echo "The virtual environment has already been activated."
    fi
}

# Функция установки зависимостей из requirements.txt
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo "File requirements.txt not found."
        return 1
    fi

    # Проверка всех зависимостей из requirements.txt
    for package in $(cat requirements.txt | cut -d '=' -f 1); do
        if ! pip freeze | grep -q "^$package=="; then
            echo "Dependency installation..."
            pip install -r requirements.txt
            echo "Dependencies installed."
            return 0
        fi
    done

    echo "All dependencies are already installed."
}

# Создание среды, если она не создана
if [ ! -d ".venv" ]; then
    create_venv > result.txt
fi

# Активация среды
activate_venv >> result.txt

# установка зависимостей
install_deps >> result.txt

# Запуск data creation script
python python_script/data_creation.py >> result.txt

# Запуск the data preprocessing script
python python_script/model_preprocessing.py >> result.txt

# Запуск the model preparation and training script
python python_script/model_preparation.py >> result.txt

# Запуск the model testing script
python python_script/model_testing.py >> result.txt
