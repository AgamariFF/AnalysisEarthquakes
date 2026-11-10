# 🌍 Анализ сейсмической активности

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.0+-orange.svg)](https://spark.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-HDFS-green.svg)](https://hadoop.apache.org/)
[![MLlib](https://img.shields.io/badge/MLlib-Spark-red.svg)](https://spark.apache.org/mllib/)
[![License](https://img.shields.io/badge/License-CC0-yellow.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

> Комплексный анализ сейсмической активности Земли с применением технологий Big Data и машинного обучения. Прогнозирование землетрясений на 2026-2030 годы, классификация по глубине очага и магнитуде.

---

## 📋 О проекте

**Цель:** Комплексный анализ сейсмической активности Земли и выявление основных закономерностей в распределении землетрясений по времени, магнитуде, глубине очага и географическому положению.

**Ключевые задачи:**
- 📊 Статистический анализ 1M+ сейсмических событий за 126 лет
- 🔬 Применение ML для классификации и прогнозирования
- 🗺️ Выявление пространственно-временных закономерностей
- 📈 Создание прогнозных моделей на 2026-2030 годы

**Практическая значимость:**
- Оценка сейсмических рисков для строительства и страхования
- Актуализация карт сейсмического районирования России
- Прогнозирование для Камчатки, Курил, Байкала и Северного Кавказа

---

## 🛠️ Технологический стек

| Категория | Технология | Назначение |
|-----------|------------|------------|
| **Хранилище** | Hadoop HDFS | Распределённое хранение данных |
| **Обработка** | Apache Spark / PySpark | Распределённые вычисления |
| **ML** | Spark MLlib | Машинное обучение |
| **Визуализация** | Power BI | Интерактивные дашборды |
| **Интерфейс** | Hue | Веб-интерфейс для Hadoop |

---

## 📊 Датасет

**Источник:** [Global Earthquake History: 1M+ Since 1900](https://www.kaggle.com/datasets/darkmatternet/global-earthquake-history-1m-since-1900)

| Параметр | Значение |
|----------|----------|
| **Записей** | 1 054 000+ |
| **Период** | 1900 — 2026 |
| **Магнитуда** | ≥ 2.5 |
| **Размер** | 228.62 МБ |
| **Лицензия** | CC0 (Public Domain) |

**Сейсмические зоны:**
- 🔥 Тихоокеанское огненное кольцо (Ring of Fire)
- 🌊 Срединно-Атлантический хребет
- 🏔️ Альпийско-Гималайский пояс

**Ключевые поля:** `time`, `year`, `latitude`, `longitude`, `depth`, `mag`, `magType`, `place`, `tsunami`, `mag_category`, `depth_category`

---

## 📁 Структура проекта

```
AnalysisEarthquakes/
├── README.md                                    # Документация проекта
├── project_analysis.md                          # Детальный разбор кода
├── dataset/
│   ├── dataset_description.md                   # Описание датасета
│   └── earthquakes_1900_2026.csv               # Данные (228 МБ)
├── src/
│   ├── forecast_number_earthquakes_world2026_2030.py
│   ├── depth_forecast_2026_2030.py
│   ├── analysis_and_forecast_number_strong_earthquakes.py
│   └── Forecast_proportion_earthquakes_magnitude_Russia.py
└── presentation/
    ├── Analiz-sejsmicheskoj-aktivnosti.pptx
    ├── presentation_summary.md
    └── presentation_text.md                  # Результаты презентации
```

---

## 🔬 ML-модели и задачи

| Скрипт | Задача | Модели | Результат |
|--------|--------|--------|-----------|
| `forecast_number_earthquakes_world2026_2030.py` | Прогноз мирового числа землетрясений | LinearRegression, RandomForestRegressor, GBTRegressor | Прогноз на 2026-2030 гг. |
| `depth_forecast_2026_2030.py` | Классификация глубины очага | LogisticRegression, RandomForestClassifier | F1 > 0.85 для intermediate |
| `analysis_and_forecast_number_strong_earthquakes.py` | Анализ сильных землетрясений (M>7) | LinearRegression с временными признаками | Прогноз интервала между M>8 |
| `Forecast_proportion_earthquakes_magnitude_Russia.py` | Прогноз по магнитуде для России | RandomForestClassifier | Классификация по 9 категориям |

**Используемые техники:**
- Lag-признаки (lag_1, lag_2, ..., lag_6)
- Скользящие средние (rolling_mean_3, rolling_mean_5, rolling_mean_7)
- Циклическое кодирование (month_sin, month_cos)
- Географические бины (lat_bin, lon_bin)
- Балансировка классов (classWeight)

---

## 🎯 Ключевые результаты

### 1. Подтверждение закона Гутенберга-Рихтера
Логарифмическая зависимость подтверждена на данных за 126 лет:
```
log₁₀ N = a - bM
```
На одно землетрясение с M≥6 приходится ~10 событий с M≥5 и ~100 с M≥4.

### 2. Классификация по глубине очага

| Категория | Глубина | Доля | Особенности |
|-----------|---------|------|-------------|
| **Shallow** | 0-70 км | ~80% | Хрупкие деформации коры, наиболее опасны |
| **Intermediate** | 70-300 км | ~17% | Зоны субдукции, фазовые переходы |
| **Deep** | 300-700 км | ~3% | Пластические деформации в мантии |

### 3. Прогнозирование на 2026-2030
- Итеративный прогноз с использованием предыдущих предсказаний
- Минимизация RMSE через ансамблевые методы
- Учёт временных трендов и сезонности

### 4. Топ-20 сейсмических зон
Выделены 20 зон с максимальной сейсмической активностью (Тихоокеанское огненное кольцо).

---

## 👥 Авторы

**Команда проекта:** Студенты курса Big Data Samsung Innovation Campus

- **Деняк Данил Евгеньевич**
- **Загайнов Владислав Евгеньевич**
- **Салий Максим Олегович**

📅 **Дата защиты:** 24.06.2026, Ростов-на-Дону

---

## 🔗 Ссылки

- [📊 Датасет на Kaggle](https://www.kaggle.com/datasets/darkmatternet/global-earthquake-history-1m-since-1900)
- [📁 Презентация проекта](./presentation/Analiz-sejsmicheskoj-aktivnosti.pptx)
- [🔬 Детальный анализ кода](./project_analysis.md)
- [🌐 Samsung Innovation Campus](https://innovationcampus.ru/)

---

## 🚀 Запуск

**Требования:**
- Apache Spark 3.0+
- Hadoop HDFS
- Python 3.8+ с PySpark

**Пример запуска:**
```bash
# Загрузка данных в HDFS
hdfs dfs -put dataset/earthquakes_1900_2026.csv /user/student/

# Запуск анализа
spark-submit src/forecast_number_earthquakes_world2026_2030.py
```

---

## 📜 Лицензия

Датасет распространяется под лицензией **CC0 (Public Domain)**. Код проекта доступен для образовательных и исследовательских целей.

---

<div align="center">

**⭐ Если проект был полезен, поставьте звезду! ⭐**

Made with ❤️ by Big Data Team | Samsung Innovation Campus 2026

</div>
