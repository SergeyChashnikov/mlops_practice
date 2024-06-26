# Practices for MLops

Study project

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

* Разработан собственный конвейер автоматизации для проекта машинного обучения. Конвеер запущен на виртуальной машине с установленным Jenkins, python и необходимыми библиотеками. Автоматизированы: сбор данных, подготовка датасета, обучение модели и работа модели.
* Разработанный конвеер выгружен в файл. Так же все скрипты (этапы конвеера сохранены)
* Все файлы размещены в подкаталоге lab2 корневого каталога

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

В этом задании необходимо развернуть микросервис в контейнере докер.
В данном варианте, это будет приложение развернутое с помощью streamlit. 

Этапы задания:
* Подготовить python код для модели и микросервиса
* Создать Docker file
* Создать docker образ
* Запустить docker контейнер и проверить его работу

</details>

## Module 4
<details>

В практическом задании данного модуля необходимо продемонстрировать навыки практического использования утилиты dvc для работы с данными.

Этапы задания:

* Создаем папку lab4 в корне проекта.
* Устанавливаем git и dvc. Настраиваем папку проекта для работы с git и dvc.
* Настраиваем удаленное хранилище файлов Google Disk.
* Создаем датасет о пассажирах “Титаника” catboost.titanic().
* Модифицируем датасет, в котором содержится информация о классе (“Pclass”), поле (“Sex”) и возрасте (“Age”) пассажира. Делаем коммит в git и push в dvc.
* Создаем новую версию датасета, в котором пропущенные (nan) значения в поле “Age” будут заполнены средним значением. Делаем коммит в git и push в dvc.
* Создаем новый признак с использованием one-hot-encoding для строкового признака “Пол” (“Sex”). Делаем коммит в git и push в dvc.
* Выполняем переключение между всеми созданными версиями датасета.
* При правильном выполнении задания у нас появится git репозиторий с опубликованной метаинформацией и папка на Google Disk, в которой хранятся различные версии датасетов. Вам необходимо подготовить отчет в тех функциональностях которые вы настроили.

</details>

## Module 5
<details>

Цель задания: применить средства автоматизации тестирования python для автоматического тестирования качества работы модели машинного обучения на различных датасетах. Результаты размещаются в каталоге lab5.

Этапы задания:

* Создать три датасета с «качественными» данными, на которых можно обучить простую модель линейной регрессии.
* На одном из этих датасетов обучить модель линейной регрессии
* Создать датасет с шумом в данных
* Провести тестирование работы модели на разных датасетах с использованием pytest, анализируя качество предсказания, обнаружить проблему на датасете с шумами.

</details>