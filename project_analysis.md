# Разбор кода проекта "Анализ сейсмической активности"

Проект представляет собой **комплексный анализ сейсмической активности** с использованием Apache Spark (PySpark). 4 Python-скрипта решают задачи машинного обучения на датасете землетрясений USGS (1+ млн событий с 1900 года).

---

## Структура проекта

```
AnalysisEarthquakes/
├── src/
│   ├── forecast_number_earthquakes_world2026_2030.py
│   ├── depth_forecast_2026_2030.py
│   ├── analysis_and_forecast_number_strong_earthquakes.py
│   └── Forecast_proportion_earthquakes_magnitude_Russia.py
├── dataset/
│   └── dataset_description.md
└── presentation/
    └── presentation_summary.md
```

---

## 1. `forecast_number_earthquakes_world2026_2030.py`

### Назначение
**Прогнозирование общего количества землетрясений в мире на 2026-2030 годы.**

### Архитектура решения

```
Данные → Агрегация по годам → Создание временных признаков → Обучение 3 моделей → Выбор лучшей → Прогноз
```

### Этапы работы

#### 1.1 Загрузка и подготовка данных (строки 7-33)
```python
df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)

# Преобразование типов
df = df.withColumn("year", col("year").cast(IntegerType()))
       .withColumn("latitude", col("latitude").cast(DoubleType()))
       ...
```

- Читает CSV из HDFS (`/user/student/earth.csv`)
- `inferSchema=True` — Spark автоматически определяет типы колонок
- Явное приведение типов для надёжности

#### 1.2 Агрегация по годам (строки 35-43)
```python
yearly = df_train_source.groupBy("year").agg(
    count("*").alias("earthquake_count"),
    avg("mag").alias("avg_mag"),
    spark_max("mag").alias("max_mag"),
    avg("depth").alias("avg_depth"),
    count(when(col("mag") >= 4.5, True)).alias("strong_quakes_45plus"),
    count(when(col("mag") >= 5.0, True)).alias("strong_quakes_5plus"),
    count(when(col("mag") >= 6.0, True)).alias("strong_quakes_6plus")
).orderBy("year")
```

**Что получается:**
| year | earthquake_count | avg_mag | max_mag | strong_quakes_5plus |
|------|------------------|---------|---------|---------------------|
| 1900 | 1523 | 5.2 | 8.3 | 45 |
| 1901 | 1601 | 5.1 | 7.8 | 38 |
| ... | ... | ... | ... | ... |

#### 1.3 Создание временных признаков (строки 44-46)

**Ключевая часть ML-подготовки** — создание признаков для временного ряда:

```python
w = Window.orderBy("year")

yearly_features = yearly
    # Lag-признаки (значения за прошлые годы)
    .withColumn("lag_1", lag("earthquake_count", 1).over(w))   # 1 год назад
    .withColumn("lag_2", lag("earthquake_count", 2).over(w))   # 2 года назад
    .withColumn("lag_3", lag("earthquake_count", 3).over(w))
    .withColumn("lag_4", lag("earthquake_count", 4).over(w))
    .withColumn("lag_5", lag("earthquake_count", 5).over(w))
    .withColumn("lag_6", lag("earthquake_count", 6).over(w))

    # Скользящие средние (сглаживание шума)
    .withColumn("rolling_mean_3", avg("earthquake_count").over(w.rowsBetween(-2, 0)))
    .withColumn("rolling_mean_5", avg("earthquake_count").over(w.rowsBetween(-4, 0)))
    .withColumn("rolling_mean_7", avg("earthquake_count").over(w.rowsBetween(-6, 0)))

    # Тренды (учёт нелинейности)
    .withColumn("trend", col("year") - lit(1900))
    .withColumn("trend_sq", pow(col("year") - lit(1900), lit(2)))

    # Разности (изменения между годами)
    .withColumn("diff_1", col("earthquake_count") - col("lag_1"))
    .withColumn("diff_2", col("lag_1") - col("lag_2"))

    # Признаки магнитуды и глубины
    .withColumn("mag_lag_1", lag("avg_mag", 1).over(w))
    .withColumn("mag_lag_2", lag("avg_mag", 2).over(w))
    .withColumn("depth_lag_1", lag("avg_depth", 1).over(w))
    .withColumn("depth_lag_2", lag("avg_depth", 2).over(w))
    ...
```

**Типы признаков:**

| Тип | Примеры | Зачем нужны |
|-----|---------|-------------|
| **Lag-признаки** | lag_1, lag_2, ..., lag_6 | Модель "видит" историю — значение в этом году зависит от предыдущих |
| **Скользящие средние** | rolling_mean_3, rolling_mean_5, rolling_mean_7 | Сглаживают шум, выявляют тренд |
| **Тренды** | trend, trend_sq | Учитывают линейный и квадратичный рост |
| **Разности** | diff_1, diff_2 | Модель учитывает скорость изменения |
| **Внешние признаки** | mag_lag_1, depth_lag_1 | Корреляция между магнитудой/глубиной и числом событий |

#### 1.4 Разделение данных (строки 63-65)

```python
train = yearly_features.filter(col("year") <= 2015)    # Обучение
val = yearly_features.filter((col("year") >= 2016) & (col("year") <= 2019))   # Валидация
test = yearly_features.filter((col("year") >= 2020) & (col("year") <= 2025))  # Тест
```

**Важно:** Данные **не перемешиваются** (нет `randomSplit` для всей выборки).

**Почему:** Это временной ряд. Если перемешать, модель будет "видеть будущее" при обучении → data leakage.

**Правильное разделение для временных рядов:**
```
[==== TRAIN ====][= VAL =][== TEST ==][ FUTURE ]
   1900-2015      2016-19   2020-2025   2026+
```

#### 1.5 Масштабирование признаков (строки 70-74)

```python
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,    # Деление на стандартное отклонение
    withMean=True    # Вычитание среднего
)
scaler_model = scaler.fit(train_assembled)
train_scaled = scaler_model.transform(train_assembled)
```

**Зачем:** LinearRegression чувствителен к масштабу. Если один признак в диапазоне [0, 100], а другой [0, 0.01], первый будет доминировать.

**StandardScaler приводит все признаки к μ=0, σ=1.**

RandomForest и GBT **не требуют** масштабирования (деревья нечувствительны к масштабу).

#### 1.6 Обучение трёх моделей (строки 81-135)

| Модель | Параметры | Особенности | Когда лучше |
|--------|-----------|-------------|-------------|
| **LinearRegression** | regParam=0.1, elasticNetParam=0.2 | Линейная зависимость, работает на scaled_features | Когда зависимость близка к линейной |
| **RandomForestRegressor** | 300 деревьев, maxDepth=8 | Ансамбль, устойчив к выбросам, не требует масштабирования | Нелинейные зависимости, шумные данные |
| **GBTRegressor** | 120 итераций, maxDepth=5, stepSize=0.05 | Градиентный бустинг, последовательное улучшение | Требует точности, но медленнее обучается |

**LinearRegression:**
```python
lr = LinearRegression(
    featuresCol="scaled_features",  # Обязательно масштабированные!
    labelCol=label_col,
    regParam=0.1,           # L2-регуляризация (Ridge)
    elasticNetParam=0.2     # Mix L1/L2: 0.2 = 20% L1 + 80% L2
)
```

**RandomForestRegressor:**
```python
rf = RandomForestRegressor(
    featuresCol="features",  # Без масштабирования
    labelCol=label_col,
    numTrees=300,     # Количество деревьев
    maxDepth=8,       # Глубина дерева (сложность модели)
    minInstancesPerNode=1,  # Минимум объектов в листе
    seed=42           # Воспроизводимость
)
```

**GBTRegressor (Gradient Boosted Trees):**
```python
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=label_col,
    maxIter=120,      # Количество итераций бустинга
    maxDepth=5,       # Меньшая глубина, чем у RF (boosting переобучается)
    stepSize=0.05     # Learning rate — шаг обучения
)
```

#### 1.7 Выбор лучшей модели (строки 136-138)

```python
models_info = sorted(models_info, key=lambda x: x["rmse"])
best = models_info[0]
```

Критерий: **минимальный RMSE** (Root Mean Squared Error) на валидации.

#### 1.8 Финальное обучение (строки 139-176)

```python
train_val = yearly_features.filter(col("year") <= 2019)  # train + val
train_val_assembled = assembler.transform(train_val)

if best["name"] == "LinearRegression":
    final_model = LinearRegression(...).fit(train_val_scaled)
elif best["name"] == "RandomForest":
    final_model = RandomForestRegressor(...).fit(train_val_assembled)
else:
    final_model = GBTRegressor(...).fit(train_val_assembled)
```

**Логика:** После выбора модели на валидации, обучаем её на **train + val** — больше данных → лучше качество.

#### 1.9 Прогнозирование будущего (строки 200-275)

```python
def build_future_feature_row(future_year, series):
    prev1 = series[-1]  # Последний известный год
    prev2 = series[-2]
    ...

    lag_1 = prev1["earthquake_count"]
    lag_2 = prev2["earthquake_count"]
    ...

    rolling_mean_3 = (prev1["earthquake_count"] + prev2["earthquake_count"] + prev3["earthquake_count"]) / 3.0

    trend = future_year - 1900
    trend_sq = float(trend ** 2)

    return (future_year, trend, trend_sq, lag_1, lag_2, ...)


future_years = [2026, 2027, 2028, 2029, 2030]
future_predictions = []

for future_year in future_years:
    feature_row = build_future_feature_row(future_year, series)
    future_input = spark.createDataFrame([feature_row], schema=feature_cols)

    pred_value = final_model.transform(future_ready).collect()[0]["prediction"]

    if pred_value < 0:
        pred_value = 0.0  # Землетрясений не может быть отрицательно

    future_predictions.append((future_year, pred_value))

    # Добавляем прогноз в историю для следующего года!
    series.append({
        "year": future_year,
        "earthquake_count": pred_value,
        ...
    })
```

**Итеративный прогноз:**
1. Прогноз для 2026 строится на исторических данных до 2025
2. Прогноз 2026 добавляется в `series`
3. Прогноз для 2027 использует прогноз 2026 как lag_1
4. И так далее...

```
2026: [hist_2025, hist_2024, ...] → pred_2026
2027: [pred_2026, hist_2025, ...] → pred_2027
2028: [pred_2027, pred_2026, ...] → pred_2028
```

---

## 2. `depth_forecast_2026_2030.py`

### Назначение
**Классификация землетрясений по глубине очага и прогноз распределения по категориям на 2026-2030.**

### Категории глубины

| Категория | Глубина | Доля | Особенности |
|-----------|---------|------|-------------|
| **Shallow** | 0-70 км | ~80% | Хрупкие деформации коры, наиболее опасны |
| **Intermediate** | 70-300 км | ~17% | Зоны субдукции, фазовые переходы минералов |
| **Deep** | 300-700 км | ~3% | Пластические деформации в мантии |

### Этапы работы

#### 2.1 Подготовка данных (строки 21-55)

```python
valid_depths = [
    "shallow (0-70km)",
    "intermediate (70-300km)",
    "deep (300-700km)"
]

df = df.filter(col("depth_category_norm").isin(valid_depths))
df = df.dropna(subset=["year", "depth_category_norm", "latitude", "longitude", "depth", "mag"])
```

**Фильтрация:**
- Только корректные категории глубины
- Удаление строк с пропусками в ключевых колонках

#### 2.2 Разделение train/test (строки 60-61)

```python
train_df_full = df.filter(col("year") <= 2022)
test_df = df.filter(col("year") > 2022)
```

**Семплирование train:**
```python
train_df = train_df_full.sample(withReplacement=False, fraction=0.25, seed=42)
```

**Зачем семплирование:** Полный train слишком большой → долгое обучение. 25% достаточно для обучения с учётом миллионов записей.

#### 2.3 Кодирование целевой переменной (строки 71-81)

```python
label_indexer = StringIndexer(
    inputCol="depth_category_norm",
    outputCol="label",
    handleInvalid="error"
).fit(train_df)

train_df = label_indexer.transform(train_df)
test_df = label_indexer.transform(test_df)

labels = label_indexer.labels  # ["shallow (0-70km)", "intermediate (70-300km)", "deep (300-700km)"]
```

**StringIndexer** преобразует строки в числа:
```
"shallow (0-70km)"      → 0.0
"intermediate (70-300km)" → 1.0
"deep (300-700km)"      → 2.0
```

#### 2.4 Балансировка классов (строки 83-96)

**Проблема:** Классы несбалансированы (shallow в 27 раз чаще deep).

**Решение:** Взвешивание классов.

```python
class_counts = train_df.groupBy("label").count()
total_count = train_df.count()
num_classes = class_counts.count()

class_weights = class_counts.withColumn(
    "classWeight",
    F.lit(float(total_count)) / (F.col("count") * F.lit(float(num_classes)))
)

train_df = train_df.join(class_weights, on="label", how="left")
```

**Формула веса:**
```
weight(class) = total_count / (count(class) * num_classes)
```

**Пример:**
- total = 100000
- shallow = 80000, intermediate = 17000, deep = 3000
- weight_shallow = 100000 / (80000 * 3) = 0.42
- weight_intermediate = 100000 / (17000 * 3) = 1.96
- weight_deep = 100000 / (3000 * 3) = 11.11

**Результат:** Редкий класс (deep) получает в 26 раз больший вес → модель штрафуется сильнее за ошибки на редком классе.

#### 2.5 Pipeline для признаков (строки 106-140)

```python
stages = []

# 1. Imputer — заполнение пропусков
if numeric_feature_cols:
    imputer = Imputer(
        inputCols=numeric_feature_cols,
        outputCols=[f"{c}_imp" for c in numeric_feature_cols]
    ).setStrategy("median")  # Медиана устойчива к выбросам
    stages.append(imputer)
    numeric_out = [f"{c}_imp" for c in numeric_feature_cols]

# 2. One-Hot Encoding для категориальных признаков
for c in categorical_cols:
    idx_col = f"{c}_idx"
    ohe_col = f"{c}_ohe"

    stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
    stages.append(OneHotEncoder(inputCols=[idx_col], outputCols=[ohe_col], handleInvalid="keep"))
    encoded_cols.append(ohe_col)

# 3. VectorAssembler — сборка вектора признаков
assembler = VectorAssembler(
    inputCols=numeric_out + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)
stages.append(assembler)
```

**Pipeline stages:**
```
[numeric_cols] → Imputer → [numeric_cols_imp]
[categorical_cols] → StringIndexer → [col_idx] → OneHotEncoder → [col_ohe]
[numeric_cols_imp + col_ohe] → VectorAssembler → features
```

**One-Hot Encoding:**
```
magType = "mw"
→ StringIndexer: magType_idx = 2.0
→ OneHotEncoder: magType_ohe = [0, 0, 1, 0, 0, ...]  (одна 1, остальные 0)
```

#### 2.6 Обучение двух классификаторов (строки 142-178)

**LogisticRegression:**
```python
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",  # Балансировка!
    family="multinomial",     # Многоклассовая классификация
    maxIter=80,
    regParam=0.02,            # L2-регуляризация
    elasticNetParam=0.0,      # Чистый Ridge
    standardization=True      # Внутреннее масштабирование
)
```

**RandomForestClassifier:**
```python
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",  # Балансировка!
    numTrees=80,
    maxDepth=8,
    maxBins=32,
    minInstancesPerNode=10,   # Минимум объектов в листе
    featureSubsetStrategy="sqrt"  # √n признаков для каждого дерева
)
```

**featureSubsetStrategy="sqrt":** Каждое дерево использует только √n случайных признаков → снижает корреляцию между деревьями → лучше ансамбль.

#### 2.7 Метрики по каждому классу (строки 201-231)

```python
def per_class_metrics(pred_df, labels, model_name):
    rows = []
    for i, lbl in enumerate(labels):
        precision_val = metric_by_label(pred_df, "precisionByLabel", i)
        recall_val = metric_by_label(pred_df, "recallByLabel", i)
        f1_val = metric_by_label(pred_df, "fMeasureByLabel", i)
        rows.append((model_name, lbl, float(precision_val), float(recall_val), float(f1_val)))
    return spark.createDataFrame(rows, ["model_name", "class_name", "precision", "recall", "f1"])
```

**Метрики:**
- **Precision** = TP / (TP + FP) — доля правильных среди предсказанных
- **Recall** = TP / (TP + FN) — доля найденных среди реальных
- **F1** = 2 * precision * recall / (precision + recall) — гармоническое среднее

#### 2.8 Выбор лучшей модели (строки 236-249)

```python
if (rf_metrics["intermediate_f1"] > lr_metrics["intermediate_f1"]) or (
    rf_metrics["intermediate_f1"] == lr_metrics["intermediate_f1"] and
    rf_metrics["weighted_f1"] >= lr_metrics["weighted_f1"]
):
    best_model = rf_model
else:
    best_model = lr_model
```

**Критерий:** F1 для **intermediate** категории.

**Почему:** Intermediate — самая сложная категория:
- Shallow — легко (80% данных)
- Deep — легко (мало данных, но чёткие признаки: большая глубина)
- Intermediate — сложно (средняя частота, перекрывающиеся признаки)

#### 2.9 Прогноз распределения глубин (строки 311-389)

```python
# Прогноз общего числа землетрясений из другого скрипта
forecast_total_df = spark.createDataFrame([
    (2026, 29937.88181578379),
    (2027, 31148.5507830383),
    ...
], ["year", "predicted_earthquake_count"])

# Берём примеры из последних лет
base_sample = recent_base.select(*feature_cols).limit(sample_limit)

for row in forecast_total_df.collect():
    target_year = int(row["year"])
    target_total = float(row["predicted_earthquake_count"])

    # Меняем год в примерах
    base_year = base_sample.withColumn("year", lit(float(target_year)))

    # Применяем модель классификации
    forecast_depth_pred = best_model.transform(base_year)
    forecast_depth_pred = label_converter.transform(forecast_depth_pred)

    # Считаем долю каждой категории
    predicted_distribution = forecast_depth_pred.groupBy("predicted_depth_category").agg(
        count("*").alias("sample_predicted_count")
    )

    # Нормируем на общее число землетрясений
    predicted_distribution = predicted_distribution.withColumn(
        "forecast_events",
        F.col("predicted_share") * F.lit(target_total)
    )
```

**Алгоритм:**
1. Взять примеры землетрясений из последних лет (2020-2025)
2. Заменить год на 2026, 2027, ...
3. Применить классификатор → получить предсказанные категории глубины
4. Посчитать долю каждой категории
5. Умножить долю на прогноз общего числа землетрясений

---

## 3. `analysis_and_forecast_number_strong_earthquakes.py`

### Назначение
**Анализ и прогнозирование сильных землетрясений (M>7.0) и катастрофических (M>8.0).**

### Этапы работы

#### 3.1 Создание бинарных признаков (строка 48)

```python
df1 = df1.withColumn("is_strong", F.when(F.col("mag") > 7.0, 1).otherwise(0))
         .withColumn("is_catastrophic", F.when(F.col("mag") > 8.0, 1).otherwise(0))
```

#### 3.2 Географическая сетка (строки 64-73)

```python
df1 = df1.withColumn("lat_bin", (F.floor(F.col("latitude") / 5) * 5).cast("int"))
         .withColumn("lon_bin", (F.floor(F.col("longitude") / 5) * 5).cast("int"))

geo_grid = df1.groupBy("lat_bin", "lon_bin").agg(
    F.count("*").alias("quake_count"),
    F.avg("mag").alias("avg_mag"),
    F.max("mag").alias("max_mag"),
    F.sum("is_strong").alias("strong_count"),
    F.sum("is_catastrophic").alias("catastrophic_count")
).orderBy(F.col("quake_count").desc())

geo_grid20 = geo_grid.limit(20)  # Топ-20 зон
```

**Сетка 5°×5°:**
```
lat_bin | lon_bin | quake_count | max_mag | catastrophic_count
--------|---------|-------------|---------|-------------------
-30     | -70     | 15234       | 9.5     | 3
-15     | 170     | 12891       | 8.2     | 1
...
```

**Топ-20 зон** — где чаще всего происходят землетрясения (Тихоокеанское огненное кольцо).

#### 3.3 Анализ интервалов между катастрофами (строки 87-99)

```python
cat_df = df1.filter(F.col("mag") > 8.0)  # Только катастрофические

w = Window.orderBy("time_ts")

cat_intervals = cat_df.withColumn("prev_time", F.lag("time_ts").over(w))
     .withColumn("days_since_prev", F.datediff(F.col("time_ts"), F.col("prev_time")))
     .withColumn("years_since_prev", F.col("days_since_prev") / F.lit(365.25))

cat_stats = cat_intervals.filter(F.col("days_since_prev").isNotNull()).agg(
    F.count("*").alias("interval_count"),
    F.avg("days_since_prev").alias("avg_days_between_cat"),
    F.expr("percentile_approx(days_since_prev, 0.5)").alias("median_days_between_cat"),
    F.min("days_since_prev").alias("min_days_between_cat"),
    F.max("days_since_prev").alias("max_days_between_cat")
)
```

**Результат:**
```
interval_count | avg_days_between_cat | median_days_between_cat | min | max
----------------|----------------------|-------------------------|-----|-----
45              | 892                  | 456                     | 12  | 3245
```

#### 3.4 Прогноз даты следующей катастрофы (строки 110-125)

```python
last_cat = cat_df.agg(F.max("time_ts").alias("last_cat_time"))
last_cat_time = last_cat.collect()[0]["last_cat_time"]

forecast_window = forecast_window
    .withColumn("forecast_mean_date", F.expr("date_add(last_cat_time, cast(avg_days_between_cat as int))"))
    .withColumn("forecast_median_date", F.expr("date_add(last_cat_time, cast(median_days_between_cat as int))"))
    .withColumn("forecast_earliest_date", F.expr("date_add(last_cat_time, cast(min_days_between_cat as int))"))
    .withColumn("forecast_latest_date", F.expr("date_add(last_cat_time, cast(max_days_between_cat as int))"))
```

**Пример:**
- Последнее M>8.0: 2024-03-15
- Средний интервал: 892 дня (2.4 года)
- Прогноз среднего: 2024-03-15 + 892 дня = **2026-09-01**
- Диапазон: 2024-03-15 + [12, 3245] дней = **2024-03-27 — 2033-01-10**

#### 3.5 Линейная регрессия для сильных землетрясений (строки 132-172)

```python
yearly_strong = df1.groupBy("year").agg(
    F.sum("is_strong").alias("strong_count_year")
).orderBy("year")

yearly_train = yearly_strong.withColumn("year_index", F.col("year") - F.lit(1900))

assembler = VectorAssembler(inputCols=["year_index"], outputCol="features")
yearly_ml = assembler.transform(yearly_train).select(
    "features",
    F.col("strong_count_year").cast("double").alias("label"),
    "year"
)

lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100)
lr_model = lr.fit(train_df)
```

**Простая модель:** Только год как признак → линейный тренд числа сильных землетрясений.

---

## 4. `Forecast_proportion_earthquakes_magnitude_Russia.py`

### Назначение
**Прогноз преобладающего класса магнитуды для каждого месяца 2027 года в России.**

### Этапы работы

#### 4.1 Фильтрация по России (строки 21-26)

```python
df_russia = df.filter(lower(col("place")).contains("russia"))
```

#### 4.2 Дискретизация магнитуды (строки 31-42)

```python
df = df.withColumn("mag_class",
    when(col("mag") < 2.0, 1)
    .when(col("mag") < 3.0, 2)
    .when(col("mag") < 4.0, 3)
    .when(col("mag") < 5.0, 4)
    .when(col("mag") < 6.0, 5)
    .when(col("mag") < 7.0, 6)
    .when(col("mag") < 8.0, 7)
    .when(col("mag") < 9.0, 8)
    .otherwise(9)
)
```

**Классы магнитуды:**
| Класс | Диапазон | Интерпретация |
|-------|----------|---------------|
| 1 | M < 2.0 | Микро |
| 2 | 2.0 ≤ M < 3.0 | Очень слабое |
| 3 | 3.0 ≤ M < 4.0 | Слабое |
| 4 | 4.0 ≤ M < 5.0 | Умеренное |
| 5 | 5.0 ≤ M < 6.0 | Среднее |
| 6 | 6.0 ≤ M < 7.0 | Сильное |
| 7 | 7.0 ≤ M < 8.0 | Разрушительное |
| 8 | 8.0 ≤ M < 9.0 | Катастрофическое |
| 9 | M ≥ 9.0 | Величайшее |

#### 4.3 Определение целевой переменной (строки 52-61)

```python
# Считаем число событий каждого класса в месяце
monthly_class_counts = df.groupBy("year", "month", "mag_class").agg(
    count("*").alias("class_count")
)

# Для каждого месяца берём самый частый класс
w_dom = Window.partitionBy("year", "month").orderBy(
    col("class_count").desc(),  # По убыванию частоты
    col("mag_class").asc()      # При равенстве — меньший класс
)

monthly_target = monthly_class_counts.withColumn("rn", row_number().over(w_dom))
    .filter(col("rn") == 1)  # Только первая строка (доминирующий класс)
    .select("year", "month", col("mag_class").alias("target_mag_class"))
```

**Пример:**
```
year | month | mag_class | class_count
-----|-------|-----------|------------
2020 | 1     | 3         | 45         ← target (самый частый)
2020 | 1     | 2         | 32
2020 | 1     | 4         | 12
```

#### 4.4 Временные признаки (строки 65-79)

```python
w_time = Window.orderBy("year", "month")

monthly_df = monthly_df
    .withColumn("lag_event_count_1", lag("event_count", 1).over(w_time))
    .withColumn("lag_event_count_2", lag("event_count", 2).over(w_time))
    .withColumn("lag_event_count_3", lag("event_count", 3).over(w_time))
    .withColumn("lag_avg_mag_1", lag("avg_mag", 1).over(w_time))
    ...

    # Циклическое кодирование месяца
    .withColumn("month_sin", sin(col("month") * lit(2 * math.pi / 12)))
    .withColumn("month_cos", cos(col("month") * lit(2 * math.pi / 12)))
```

**Циклическое кодирование:**

Проблема: Декабрь (12) и январь (1) — соседние месяцы, но численно далеки.

Решение: Кодировать месяц через sin/cos → замыкается в круг.

```
month | month_sin | month_cos
------|-----------|----------
1     | 0.5       | 0.87
2     | 0.87      | 0.5
...
11    | -0.87     | 0.5
12    | -0.5      | 0.87
```

**Графически:**
```
        3 (0, 1)
         |
  2      |      4
(-0.87)  |   (0.87)
         |
---------+--------- 12 → (1, 0), 6 → (-1, 0)
         |
  11     |     5
(-0.87)  |   (0.87)
         |
        9 (0, -1)
```

#### 4.5 RandomForestClassifier (строки 119-130)

```python
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(stages=[label_indexer, assembler, rf])
model = pipeline.fit(train_df)
```

#### 4.6 Прогноз на 12 месяцев вперёд (строки 186-248)

```python
forecast_rows = []

for m in range(1, 13):  # Январь - Декабрь 2027
    # Берём последние известные значения
    lag_event_count_1 = float(temp.iloc[-1]["event_count"])
    lag_avg_mag_1 = float(temp.iloc[-1]["avg_mag"])
    ...

    # Строим признаки для месяца m
    future_row = spark.createDataFrame([(
        2027, m,
        future_event_count, future_avg_mag, ...,
        lag_event_count_1, lag_event_count_2, ...,
        future_month_sin, future_month_cos
    )], feature_cols)

    # Предсказываем класс и вероятности
    pred_future = model.transform(future_row)
        .withColumn("prob_array", vector_to_array(col("probability")))

    row = pred_future.select("month", "prob_array", "prediction").collect()[0]
    probs = row["prob_array"]

    # Сохраняем вероятности для каждого класса
    prob_map = {int(float(labels[i])): float(probs[i]) for i in range(len(labels))}

    result = {
        "year": 2027,
        "month": m,
        "predicted_class": int(float(labels[int(row["prediction"])])),
        "prob_mag_1": prob_map.get(1, 0.0),
        "prob_mag_2": prob_map.get(2, 0.0),
        ...
    }

    forecast_rows.append(result)

    # Добавляем прогноз в историю для следующего месяца
    hist_pdf.loc[len(hist_pdf)] = [
        2027, m,
        future_event_count, ...,
        int(result["predicted_class"]),  # ← предсказанный класс
        lag_event_count_1, ...
    ]
```

**Итеративный прогноз:**
```
Январь 2027: [hist_2026-12, hist_2026-11, ...] → pred_jan
Февраль 2027: [pred_jan, hist_2026-12, ...] → pred_feb
Март 2027: [pred_feb, pred_jan, hist_2026-12, ...] → pred_mar
...
```

---

## Ключевые концепции PySpark ML

### 1. Pipeline

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    indexer,      # StringIndexer
    encoder,      # OneHotEncoder
    assembler,    # VectorAssembler
    model         # LogisticRegression / RandomForest
])

fitted_pipeline = pipeline.fit(train_df)
predictions = fitted_pipeline.transform(test_df)
```

**Зачем:** Все преобразования (кодирование, масштабирование) запоминаются и применяются одинаково к train и test.

**Без Pipeline:**
```python
# Нужно вручную применять те же преобразования
indexer_model = indexer.fit(train_df)
train_indexed = indexer_model.transform(train_df)
test_indexed = indexer_model.transform(test_df)  # Тот же indexer_model!

encoder_model = encoder.fit(train_indexed)
train_encoded = encoder_model.transform(train_indexed)
test_encoded = encoder_model.transform(test_indexed)  # Тот же encoder_model!
...
```

**С Pipeline:**
```python
# Все автоматически
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

### 2. VectorAssembler

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["col1", "col2", "col3"],
    outputCol="features"
)

df = assembler.transform(df)
```

**До:**
```
col1 | col2 | col3
-----|------|-----
1.0  | 2.5  | 3.2
```

**После:**
```
col1 | col2 | col3 | features
-----|------|------|-----------------
1.0  | 2.5  | 3.2  | [1.0, 2.5, 3.2]
```

**Зачем:** Spark ML требует, чтобы все признаки были в одной колонке типа `Vector`. Это специфический формат для матричных операций.

### 3. Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, avg, col

w = Window.orderBy("year")

df = df.withColumn("prev_year_value", lag("value", 1).over(w))
       .withColumn("rolling_mean", avg("value").over(w.rowsBetween(-2, 0)))
```

**Lag:**
```
year | value | prev_year_value
-----|-------|----------------
2020 | 100   | null
2021 | 120   | 100
2022 | 110   | 120
2023 | 130   | 110
```

**Скользящее среднее (rowsBetween(-2, 0)):**
```
year | value | rolling_mean
-----|-------|-------------
2020 | 100   | 100.0         (только текущее)
2021 | 120   | 110.0         (100, 120)
2022 | 110   | 110.0         (100, 120, 110)
2023 | 130   | 120.0         (120, 110, 130)
```

**Суть:** Window functions позволяют делать вычисления "вдоль" временного ряда — брать предыдущие значения, скользящие средние, ранги. Это как GROUP BY, но без схлопывания строк.

### 4. Imputer

```python
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["col1", "col2"],
    outputCols=["col1_imp", "col2_imp"]
).setStrategy("median")  # или "mean"

imputer_model = imputer.fit(df)
df_filled = imputer_model.transform(df)
```

**Зачем:** Пропуски в данных. Медиана предпочтительнее среднего — устойчива к выбросам.

### 5. StringIndexer + OneHotEncoder

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Шаг 1: Строка → Число
indexer = StringIndexer(inputCol="category", outputCol="category_idx")
df_indexed = indexer.fit(df).transform(df)

# category | category_idx
#----------|-------------
# "A"      | 0.0
# "B"      | 1.0
# "C"      | 2.0
# "A"      | 0.0

# Шаг 2: Число → One-Hot вектор
encoder = OneHotEncoder(inputCols=["category_idx"], outputCols=["category_ohe"])
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# category_idx | category_ohe
#--------------|-------------
# 0.0          | [1.0, 0.0, 0.0]
# 1.0          | [0.0, 1.0, 0.0]
# 2.0          | [0.0, 0.0, 1.0]
# 0.0          | [1.0, 0.0, 0.0]
```

**Зачем два шага:** Spark ML работает только с числами. One-Hot Encoding превращает категорию в бинарный вектор.

### 6. StandardScaler

```python
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,   # Деление на std
    withMean=True   # Вычитание mean
)

scaler_model = scaler.fit(train_df)
train_scaled = scaler_model.transform(train_df)
test_scaled = scaler_model.transform(test_df)  # Тот же scaler_model!
```

**Формула:** `scaled = (x - mean) / std`

**Зачем:** LinearRegression чувствителен к масштабу. RandomForest — нет.

---

## Связь между скриптами

```
forecast_number_earthquakes_world2026_2030.py
    ↓
    Прогноз общего числа землетрясений на 2026-2030
    ↓
depth_forecast_2026_2030.py
    ↓
    Распределяет прогноз по категориям глубины
    (использует прогноз общего числа из первого скрипта)
    ↓

analysis_and_forecast_number_strong_earthquakes.py
    (Отдельный анализ M>7.0 и M>8.0)
    ↓

Forecast_proportion_earthquakes_magnitude_Russia.py
    (Отдельный анализ для России)
```

---

## Результаты

Все скрипты сохраняют результаты в HDFS:

```
/user/student/Samsung_project_output/
├── depth_category_test_predictions_final/
├── depth_category_per_class_metrics_final/
├── depth_category_yearly_share_final/
├── depth_category_forecast_2026_2030_final/
├── output_yearly/
├── output_val/
├── output_test/
├── output_forecast/
├── geo_points/
├── geo_grid/
├── catastrophic_events/
├── catastrophic_intervals/
├── catastrophic_stats/
├── catastrophic_forecast_window/
├── yearly_strong/
├── model_test_predictions/
├── model_metrics/
├── model_residual_stats/
├── monthly_forecast/
├── test_predictions/
└── metrics/
```

---

## Выводы

1. **Архитектура:** Big Data pipeline (HDFS → PySpark → MLlib → визуализация)

2. **ML-подходы:**
   - Регрессия: прогноз числа землетрясений
   - Классификация: категория глубины, класс магнитуды
   - Временные ряды: lag-признаки, скользящие средние, тренды

3. **Ключевые техники:**
   - Балансировка классов (взвешивание)
   - One-Hot Encoding для категориальных признаков
   - Циклическое кодирование для сезонности
   - Итеративный прогноз для временных рядов

4. **Практическое применение:**
   - Оценка сейсмических рисков
   - Карты сейсмического районирования
   - Прогнозирование для строительства и страхования
