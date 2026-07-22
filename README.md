# JMLC-Room-description-sufficiency
ML-решение для классификации и детекции неполных описаний отельных номеров для сервиса Т-Путешествия. Мета-модель (LightGBM+BERT) для оптимизации расходов на краудсорсинг. Метрика PR-AUC: 0.95 (Private) / 0.993 (Public).

Автоматическая классификация названий номеров отелей (`supplier_room_name → target`) с помощью ансамбля моделей: **DistilBERT + LightGBM (с LaBSE-эмбеддингами)** и мета-модели на их выходах.

Цель — **оптимизация расходов на краудсорсинг**: замена ручной разметки на предсказания модели для фильтрации некачественных описаний.

## Результаты

| Метрика | Public LB | Private LB |
|---------|-----------|------------|
| **PR-AUC** | **0.99290** | **0.95655** |

Финальная мета-модель превосходит обе базовые модели за счёт учёта согласованности предсказаний:
| Модель | PR-AUC (holdout) | ROC-AUC |
|--------|------------------|---------|
| DistilBERT solo | 0.940 | 0.944 |
| LightGBM solo | 0.952 | 0.952 |
| **Meta (BERT + LGB)** | **0.962** | **0.963** |
| Простое среднее BERT+LGB | 0.958 | — |


## Структура проекта
```
├── README.md                          # Описание проекта и инструкции
├── requirements.txt                   # Python-зависимости
├── Dockerfile                         # Образ для воспроизводимого запуска
├── docker-compose.yml                 # Удобный запуск команд
├── .dockerignore                      # Что не класть в Docker-образ
├── .gitignore                         # Что не коммитить в Git
│
├── scripts/                           # Переиспользуемые скрипты
│   ├── train_distilmbert.py           # Обучение DistilBERT
│   ├── train_lightgbm.py              # Обучение LightGBM (8-fold + LaBSE)
│   ├── train_meta_ensemble.py         # Обучение мета-модели (stacking)
│   ├── predict_lightgbm.py            # Инференс LightGBM на новых данных
│   └── predict_meta_ensemble.py       # Инференс мета-модели
│
├── data/                              # Данные (не коммитятся в Git)
│   ├── raw/
│   │   ├── public_dataset.csv         # Обучающая выборка (~184k строк)
│   │   └── submission_sample.csv      # Тест для проверки (без таргета)
│   └── artefacts/
│       ├── bert_artefacts.txt         # Ссылка на zip с BERT-весами
│       ├── lgb_artefacts.txt          # Ссылка на zip с LightGBM-моделями
│       └── meta_artefacts.txt         # Ссылка на zip с мета-моделью
│
├── notebooks/                         # Исследовательские ноутбуки
│   ├── eda_t_bank.ipynb               # EDA и визуализация данных
│   ├── BERT_model.ipynb               # Обучение DistilBERT (финальная версия)
│   ├── LightGBM_model.ipynb           # Обучение LightGBM (8-fold CV)
│   ├── CatBoost_model.ipynb           # Эксперименты: CatBoost + BERT
│   ├── ensemble exp.ipynb             # Эксперименты с ансамблями
│   └── bert-lgb.ipynb                 # Финальная мета-модель (stacking)
```

# Архитектура решения

## Базовая модель 1: DistilBERT (multilingual)
- Модель: distilbert-base-multilingual-cased
- Вход: нормализованный текст supplier_room_name (max_len=64)
- Обучение: 9 эпох, AdamW, lr=2e-5, AMP (bf16/fp16)
- Особенности: кастомный Collator, поиск порога по precision≥0.90

## Базовая модель 2: LightGBM + LaBSE
- Фичи: 15 табличных фич (длина, языки, ключевые слова, hotel-статистики) + 768-мерные LaBSE-эмбеддинги
- Обучение: 8-fold CV, early stopping по PR-AUC
- Hotel-фичи: hotel_positive_rate, hotel_room_count, hotel_target_std (считаются без лика внутри каждого фолда)

## Мета-модель: LightGBM-стекинг
- Входы: bert_prob, lgb_prob, bert × lgb, |bert − lgb|
- Обучение: 5-fold CV на holdout-выходах базовых моделей
- Идея: модель учится доверять той базовой модели, которая в конкретной ситуации увереннее


# Быстрый старт

## 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

## Скачивание артефактов
В папке data/artefacts/ лежат .txt-файлы со ссылками на zip-архивы с обученными моделями. Скачайте и распакуйте их в models/:

## Проверка модели (инференс)
Самый быстрый способ — сразу получить submission на submission_sample.csv:
python scripts/predict_meta_ensemble.py \
    --meta-model-path    models/meta_ensemble/meta_lgb_model.txt \
    --bert-test-probs    models/distilmbert/test_bert_probs.npy \
    --lgb-test-probs     models/lightgbm_labse/test_lgb_probs.npy \
    --test-csv           data/raw/submission_sample.csv \
    --output-path        submissions/meta_ensemble_submission.csv


# Если хотите обучить всё с нуля:
## Обучение DistilBERT
python scripts/train_distilmbert.py \
    --train-path   data/raw/public_dataset.csv \
    --test-path    data/raw/submission_sample.csv \
    --output-dir   models/distilmbert \
    --epochs       9

На выходе:
- models/distilmbert/best_model.pt — веса лучшей модели
- models/distilmbert/test_bert_probs.npy — вероятности на тесте
- models/distilmbert/split_indices.npz — индексы сплита

## Обучение LightGBM (8-fold + LaBSE)

python scripts/train_lightgbm.py \
    --train-path    data/raw/public_dataset.csv \
    --test-path     data/raw/submission_sample.csv \
    --output-dir    models/lightgbm_labse \
    --n-folds       8 \
    --epochs        3000

На выходе:
- models/lightgbm_labse/lgb_fold_*.txt — 8 моделей-фолдов
- models/lightgbm_labse/test_lgb_probs.npy — усреднённые вероятности на тесте
- models/lightgbm_labse/oof_lgb_probs.npy — OOF-предсказания для мета-модели

## Обучение мета-модели (stacking)
python scripts/train_meta_ensemble.py \
    --bert-holdout-probs  models/distilmbert/oof_bert_probs.npy \
    --bert-test-probs     models/distilmbert/test_bert_probs.npy \
    --lgb-holdout-probs   models/lightgbm_labse/oof_lgb_probs.npy \
    --lgb-test-probs      models/lightgbm_labse/test_lgb_probs.npy \
    --holdout-csv         data/raw/public_dataset.csv \
    --test-csv            data/raw/submission_sample.csv \
    --output-dir          models/meta_ensemble

На выходе:
- models/meta_ensemble/meta_lgb_model.txt — финальная мета-модель
- models/meta_ensemble/submission.csv — итоговый submission

## Инференс на новых данных

Python scripts/predict_meta_ensemble.py \
    --meta-model-path    models/meta_ensemble/meta_lgb_model.txt \
    --bert-test-probs    <новые_вероятности_BERT.npy> \
    --lgb-test-probs     <новые_вероятности_LGB.npy> \
    --test-csv           <новый_тест.csv> \
    --output-path        submissions/new_submission.csv
