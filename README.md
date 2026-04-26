# HW5: Обеспечение воспроизводимости эксперимента (MLOps)

## Цель проекта

Цель проекта — построить минимальный MLOps-пайплайн, обеспечивающий:
- воспроизводимость экспериментов
- версионирование данных и моделей
- логирование экспериментов
- централизованное управление признаками

В рамках проекта используются инструменты: **DVC, MLflow, Feast**.

---

## Структура проекта

```
mlops_hw5/
│
├── data/
│ ├── raw/ # исходные данные (через DVC)
│ └── processed/ # подготовленные данные
│
├── src/
│ ├── prepare.py # подготовка данных
│ └── train.py # обучение модели
│
├── local_repo/
│ └── feature_repo/ # Feature Store (Feast)
│
├── dvc.yaml # описание пайплайна
├── params.yaml # гиперпараметры
├── requirements.txt # зависимости
├── README.md # документация

```

## Как запустить проект

### 1. Установить зависимости

```bash
pip install -r requirements.txt

dvc pull
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

В проекте настроено хранилище признаков:

- используется PostgreSQL
- описан feature_store.yaml
- применён через feast apply

```bash
feast entities list
feast feature-views list
```