# mlops_practice
study project

# Practices for MLops course UrFU + SkillFactory
## Module 1
<details>

* Простейший конвейр для автоматизации работы с моделью машинного обучения. 
* Отдельные этапы конвейера машинного обучения описываются в разных python–скриптах, которые потом соединяются в единую цепочку действий с помощью bash-скрипта.
* Все файлы размещены в подкаталоге lab1 корневого каталога

Этапы:
1. Python-скрипт (data_creation.py), загружает набор данных, описывающие рынок IT вакансий. Данный набор скачивается из репозитория GitHub https://raw.githubusercontent.com/SergeyChashnikov/URFUML2023_STUDIES/main/MOMO/1_semestr/Salary_Data/Salary_Data_clear.csv, скрипт разделяет данные на тестовые и тренировочные и сохраняется в папки data/test и data/train.
2. Python-скрипт (data_preprocessing.py), выполняет предобработку данных, с помощью sklearn.preprocessing.StandardScaler, sklearn.preprocessing.OrdinalEncoder. Трансформации выполняются и над тестовой и над обучающей выборкой. Сохраняется в папки data/test и data/train.
3. Python-скрипт (model_preparation.py), создает и обучает модель машинного обучения на построенных данных из папки “train”. Сохраняет модель в файл, в папку model/
4. Python-скрипт (model_testing.py), проверяет модель машинного обучения на построенных данных из папки “test”.
5. Bash-скрипт (pipeline.sh), последовательно запускает все python-скрипты. В результате выполнения скрипта на терминал в стандартный поток вывода печатается одна строка с оценкой метрики модели:

```shell
Model r2-score:   0.796882707227601
```

</details>

## Module 2
<details>

* Разработан собственный конвейер автоматизации для проекта машинного обучения. Для этого нам понадобится виртуальная машина с установленным Jenkins, python и необходимыми библиотеками. Необходимо автоматизировать сбор данных, подготовку датасета, обучение модели и работу модели.
* Разработанный конвеер требуется выгрузить в файл. Так же все скрипты (этапы конвеера требуется сохранить)
* Все файлы необходимо разместить в подкаталоге lab2 корневого каталога

Этапы задания:
1. Разворачиваем сервер с Jenkins, устанавливаем необходимое программное обеспечение для работы над созданием модели машинного обучения.
2. Скачиваем данные из GitHub, (download_data.py).
3. Проводим обработку данных, формируем датасеты для тренировки и тестирования модели, сохраняем, (preprocess.py).
4. Создаем и обучаем на тренировочном датасете модель машинного обучения, сохраняем в pickle или аналогичном формате, (train_model.py).
5. Загружаем сохраненную модель на предыдущем этапе и анализируем ее качество на тестовых данных, (test_model.py). 
6. Реализовываем задания и конвеер. Связывааем конвеер с системой контроля версий. Сохраняем конвеер.

</details>

## Module 3
<details>

В практическом задание по модулю вам необходимо применить полученные знания по работе с docker (и docker-compose). Вам необходимо использовать полученные ранее знания по созданию микросервисов. В этом задании необходимо развернуть микросервис в контейнере докер. Например, это может быть модель машинного обучения, принимающая запрос по API и возвращающая ответ. Вариантом может быть реализация приложения на основе streamlit (https://github.com/korelin/streamlit_demo_app).
Результаты работы над этой работой стоит поместить в подкаталог lab3 вашего корневого каталога репозитория.
Что необходимо выполнить:
* Подготовить python код для модели и микросервиса
* Создать Docker file
* Создать docker образ
* Запустить docker контейнер и проверить его работу

Дополнительными плюсами будут:
1. Использование docker-compose
2. Автоматизация сборки образа привязка имени тэга к версии сборки (sha-коммита, имя ветки)
3. Деплой (загрузка) образа в хранилище артефактов например dockerhub

</details>
