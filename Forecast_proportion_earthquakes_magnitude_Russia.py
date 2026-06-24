#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lower, when, count, avg, max as spark_max, min as spark_min,
    lag, lit, lpad, concat_ws, sin, cos, row_number
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import math
file_path = "/user/student/earth.csv"

df = spark.read     .option("header", "true")     .option("inferSchema", "true")     .csv(file_path)
df_russia = df.filter(lower(col("place")).contains("russia"))

df = df_russia.select(
    "id", "time", "year", "month", "place",
    "latitude", "longitude", "depth", "mag"
)

df = df.dropna(subset=["year", "month", "depth", "mag"])
df = df.withColumn("year", col("year").cast("int"))        .withColumn("month", col("month").cast("int"))

df = df.withColumn(
    "mag_class",
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

monthly_base = df.groupBy("year", "month").agg(
    count("*").alias("event_count"),
    avg("mag").alias("avg_mag"),
    spark_max("mag").alias("max_mag"),
    spark_min("mag").alias("min_mag"),
    avg("depth").alias("avg_depth")
)

monthly_class_counts = df.groupBy("year", "month", "mag_class").agg(
    count("*").alias("class_count")
)

w_dom = Window.partitionBy("year", "month").orderBy(
    col("class_count").desc(),
    col("mag_class").asc()
)

monthly_target = monthly_class_counts     .withColumn("rn", row_number().over(w_dom))     .filter(col("rn") == 1)     .select("year", "month", col("mag_class").alias("target_mag_class"))

monthly_df = monthly_base.join(monthly_target, on=["year", "month"], how="inner")

w_time = Window.orderBy("year", "month")

monthly_df = monthly_df     .withColumn("lag_event_count_1", lag("event_count", 1).over(w_time))     .withColumn("lag_event_count_2", lag("event_count", 2).over(w_time))     .withColumn("lag_event_count_3", lag("event_count", 3).over(w_time))     .withColumn("lag_avg_mag_1", lag("avg_mag", 1).over(w_time))     .withColumn("lag_avg_mag_2", lag("avg_mag", 2).over(w_time))     .withColumn("lag_max_mag_1", lag("max_mag", 1).over(w_time))     .withColumn("lag_avg_depth_1", lag("avg_depth", 1).over(w_time))

monthly_df = monthly_df.fillna({
    "lag_event_count_1": 0,
    "lag_event_count_2": 0,
    "lag_event_count_3": 0,
    "lag_avg_mag_1": 0.0,
    "lag_avg_mag_2": 0.0,
    "lag_max_mag_1": 0.0,
    "lag_avg_depth_1": 0.0
})

monthly_df = monthly_df     .withColumn("month_sin", sin(col("month") * lit(2 * math.pi / 12)))     .withColumn("month_cos", cos(col("month") * lit(2 * math.pi / 12)))

monthly_df = monthly_df.orderBy("year", "month")

print("Monthly dataset shape:", monthly_df.count())
monthly_df.show(10, truncate=False)

train_df = monthly_df.filter(col("year") <= 1999)
test_df = monthly_df.filter((col("year") >= 2000) & (col("year") <= 2025))

print("Train months:", train_df.count())
print("Test months:", test_df.count())

print("Train target distribution:")
train_df.groupBy("target_mag_class").count().orderBy("target_mag_class").show()

print("Test target distribution:")
test_df.groupBy("target_mag_class").count().orderBy("target_mag_class").show()

feature_cols = [
    "year", "month",
    "event_count", "avg_mag", "max_mag", "min_mag", "avg_depth",
    "lag_event_count_1", "lag_event_count_2", "lag_event_count_3",
    "lag_avg_mag_1", "lag_avg_mag_2",
    "lag_max_mag_1", "lag_avg_depth_1",
    "month_sin", "month_cos"
]

label_indexer = StringIndexer(
    inputCol="target_mag_class",
    outputCol="label",
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    seed=42,
    probabilityCol="probability"
)

pipeline = Pipeline(stages=[label_indexer, assembler, rf])

model = pipeline.fit(train_df)

pred_train = model.transform(train_df)
pred_test = model.transform(test_df)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

train_accuracy = evaluator_acc.evaluate(pred_train)
train_f1 = evaluator_f1.evaluate(pred_train)

test_accuracy = evaluator_acc.evaluate(pred_test)
test_f1 = evaluator_f1.evaluate(pred_test)

print("Train Accuracy:", train_accuracy)
print("Train F1:", train_f1)
print("Test Accuracy:", test_accuracy)
print("Test F1:", test_f1)

labels = model.stages[0].labels

pred_test_result = pred_test.select(
    "year",
    "month",
    "target_mag_class",
    "label",
    "prediction"
)

pred_test_result = pred_test_result.rdd.map(
    lambda r: (
        r["year"],
        r["month"],
        int(r["target_mag_class"]),
        int(float(labels[int(r["prediction"])])) if r["prediction"] is not None else None
    )
).toDF(["year", "month", "actual_class", "predicted_class"])

print("Test predictions sample:")
pred_test_result.orderBy("year", "month").show(10, truncate=False)

print("Confusion-like table on test:")
pred_test_result.groupBy("actual_class", "predicted_class").count()     .orderBy("actual_class", "predicted_class")     .show(10, truncate=False)

hist_monthly = monthly_df.filter(col("year") <= 2025).orderBy("year", "month")
hist_pdf = hist_monthly.toPandas()

forecast_rows = []

for m in range(1, 13):
    temp = hist_pdf.sort_values(["year", "month"]).reset_index(drop=True)

    lag_event_count_1 = float(temp.iloc[-1]["event_count"]) if len(temp) >= 1 else 0.0
    lag_event_count_2 = float(temp.iloc[-2]["event_count"]) if len(temp) >= 2 else 0.0
    lag_event_count_3 = float(temp.iloc[-3]["event_count"]) if len(temp) >= 3 else 0.0
    lag_avg_mag_1 = float(temp.iloc[-1]["avg_mag"]) if len(temp) >= 1 else 0.0
    lag_avg_mag_2 = float(temp.iloc[-2]["avg_mag"]) if len(temp) >= 2 else 0.0
    lag_max_mag_1 = float(temp.iloc[-1]["max_mag"]) if len(temp) >= 1 else 0.0
    lag_avg_depth_1 = float(temp.iloc[-1]["avg_depth"]) if len(temp) >= 1 else 0.0
    future_event_count = lag_event_count_1
    future_avg_mag = lag_avg_mag_1
    future_max_mag = lag_max_mag_1
    future_min_mag = float(temp.iloc[-1]["min_mag"]) if len(temp) >= 1 else 0.0
    future_avg_depth = lag_avg_depth_1
    future_month_sin = float(math.sin(m * 2 * math.pi / 12))
    future_month_cos = float(math.cos(m * 2 * math.pi / 12))
    future_row = spark.createDataFrame([(
        2027, m,
        future_event_count, future_avg_mag, future_max_mag, future_min_mag, future_avg_depth,
        lag_event_count_1, lag_event_count_2, lag_event_count_3,
        lag_avg_mag_1, lag_avg_mag_2,
        lag_max_mag_1, lag_avg_depth_1,
        future_month_sin, future_month_cos,
        1
    )], [
        "year", "month",
        "event_count", "avg_mag", "max_mag", "min_mag", "avg_depth",
        "lag_event_count_1", "lag_event_count_2", "lag_event_count_3",
        "lag_avg_mag_1", "lag_avg_mag_2",
        "lag_max_mag_1", "lag_avg_depth_1",
        "month_sin", "month_cos",
        "target_mag_class"
    ])

    pred_future = model.transform(future_row)         .withColumn("prob_array", vector_to_array(col("probability")))

    row = pred_future.select("month", "prob_array", "prediction").collect()[0]
    probs = row["prob_array"]

    prob_map = {int(float(labels[i])): float(probs[i]) for i in range(len(labels))}
    result = {
        "year": 2027,
        "month": m,
        "predicted_class": int(float(labels[int(row["prediction"])]))
    }

    for cls in range(1, 10):
        result[f"prob_mag_{cls}"] = prob_map.get(cls, 0.0)

    forecast_rows.append(result)

    hist_pdf.loc[len(hist_pdf)] = [
        2027, m,
        future_event_count, future_avg_mag, future_max_mag, future_min_mag, future_avg_depth,
        int(result["predicted_class"]),
        lag_event_count_1, lag_event_count_2, lag_event_count_3,
        lag_avg_mag_1, lag_avg_mag_2,
        lag_max_mag_1, lag_avg_depth_1,
        future_month_sin, future_month_cos
    ]

forecast_2027_df = spark.createDataFrame(forecast_rows)

forecast_2027_df = forecast_2027_df.withColumn(
    "year_month",
    concat_ws("-", col("year"), lpad(col("month").cast("string"), 2, "0"))
)

forecast_2027_df.orderBy("month").show(5, truncate=False)
output_path = "/user/student/Samsung_project_output"

forecast_2027_df.coalesce(1).write.mode("overwrite")     .option("header", True)     .csv(output_path + "/monthly_forecast")

pred_test_result.coalesce(1).write.mode("overwrite")     .option("header", True)     .csv(output_path + "/test_predictions")

metrics_df = spark.createDataFrame([
    ("train_accuracy", float(train_accuracy)),
    ("train_f1", float(train_f1)),
    ("test_accuracy", float(test_accuracy)),
    ("test_f1", float(test_f1))
], ["metric", "value"])

metrics_df.coalesce(1).write.mode("overwrite")     .option("header", True)     .csv(output_path + "/metrics")


# In[ ]:




