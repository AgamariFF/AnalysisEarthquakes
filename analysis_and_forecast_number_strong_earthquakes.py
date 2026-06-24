#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import (
    col, when, floor, concat_ws, lpad, count, avg, max as spark_max,
    to_timestamp, year, month
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

file_path = "/user/student/earth.csv"
output_dir = "/user/student/Samsung_project_output"

df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)

df = df.select(
    "id", "time", "year", "month", "day_of_year", "hour",
    "latitude", "longitude", "depth", "mag", "magType", "tsunami"
)

df = df.dropna(subset=[
    "time", "year", "month", "day_of_year", "hour",
    "latitude", "longitude", "depth", "mag", "magType", "tsunami"
])

df1 = df.select(
    "id",
    "time",
    "year",
    "month",
    "day_of_year",
    "hour",
    "latitude",
    "longitude",
    "depth",
    "mag"
)

df1 = df1.withColumn("time_ts", F.to_timestamp(F.col("time")))          .withColumn("date", F.to_date(F.col("time_ts")))          .filter(F.col("time_ts").isNotNull())          .filter(F.col("latitude").between(-90, 90))          .filter(F.col("longitude").between(-180, 180))          .filter(F.col("depth") >= 0)          .filter(F.col("mag") > 0)

df1 = df1.withColumn("year", F.year(F.col("time_ts")))          .withColumn("month", F.month(F.col("time_ts")))          .withColumn("is_strong", F.when(F.col("mag") > 7.0, 1).otherwise(0))          .withColumn("is_catastrophic", F.when(F.col("mag") > 8.0, 1).otherwise(0))          .withColumn("lat_bin", (F.floor(F.col("latitude") / 5) * 5).cast("int"))          .withColumn("lon_bin", (F.floor(F.col("longitude") / 5) * 5).cast("int"))          .withColumn("year_month", F.concat_ws("-", F.col("year"), F.lpad(F.col("month").cast("string"), 2, "0")))          .withColumn("month_sin", F.sin(2 * 3.1415926535 * F.col("month") / F.lit(12)))          .withColumn("month_cos", F.cos(2 * 3.1415926535 * F.col("month") / F.lit(12)))

geo_points = df1.select(
    "id",
    "time_ts",
    "date",
    "year",
    "month",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "is_strong",
    "is_catastrophic"
)

geo_grid = df1.groupBy("lat_bin", "lon_bin").agg(
    F.count("*").alias("quake_count"),
    F.avg("mag").alias("avg_mag"),
    F.max("mag").alias("max_mag"),
    F.avg("depth").alias("avg_depth"),
    F.sum("is_strong").alias("strong_count"),
    F.sum("is_catastrophic").alias("catastrophic_count")
).orderBy(F.col("quake_count").desc())

geo_grid20 = geo_grid.limit(20)

cat_df = df1.filter(F.col("mag") > 8.0).select(
    "id",
    "time_ts",
    "date",
    "year",
    "month",
    "latitude",
    "longitude",
    "depth",
    "mag"
).orderBy("time_ts")

w = Window.orderBy("time_ts")

cat_intervals = cat_df.withColumn("prev_time", F.lag("time_ts").over(w))     .withColumn("prev_mag", F.lag("mag").over(w))     .withColumn("days_since_prev", F.datediff(F.col("time_ts"), F.col("prev_time")))     .withColumn("years_since_prev", F.col("days_since_prev") / F.lit(365.25))

cat_stats = cat_intervals.filter(F.col("days_since_prev").isNotNull()).agg(
    F.count("*").alias("interval_count"),
    F.avg("days_since_prev").alias("avg_days_between_cat"),
    F.expr("percentile_approx(days_since_prev, 0.5)").alias("median_days_between_cat"),
    F.min("days_since_prev").alias("min_days_between_cat"),
    F.max("days_since_prev").alias("max_days_between_cat"),
    F.avg("years_since_prev").alias("avg_years_between_cat"),
    F.expr("percentile_approx(years_since_prev, 0.5)").alias("median_years_between_cat")
)

last_cat = cat_df.agg(F.max("time_ts").alias("last_cat_time"))
last_cat_time = last_cat.collect()[0]["last_cat_time"]

stats_row = cat_stats.collect()[0]
avg_days = stats_row["avg_days_between_cat"]
median_days = stats_row["median_days_between_cat"]
min_days = stats_row["min_days_between_cat"]
max_days = stats_row["max_days_between_cat"]

forecast_window_data = [
    (
        last_cat_time,
        avg_days,
        median_days,
        min_days,
        max_days
    )
]

forecast_window = spark.createDataFrame(
    forecast_window_data,
    ["last_cat_time", "avg_days_between_cat", "median_days_between_cat", "min_days_between_cat", "max_days_between_cat"]
)

forecast_window = forecast_window     .withColumn("forecast_mean_date", F.expr("date_add(last_cat_time, cast(avg_days_between_cat as int))"))     .withColumn("forecast_median_date", F.expr("date_add(last_cat_time, cast(median_days_between_cat as int))"))     .withColumn("forecast_earliest_date", F.expr("date_add(last_cat_time, cast(min_days_between_cat as int))"))     .withColumn("forecast_latest_date", F.expr("date_add(last_cat_time, cast(max_days_between_cat as int))"))


yearly_strong = df1.groupBy("year").agg(
    F.sum("is_strong").alias("strong_count_year")
).orderBy("year")

yearly_train = yearly_strong.withColumn("year_index", F.col("year") - F.lit(1900))
assembler = VectorAssembler(inputCols=["year_index"], outputCol="features")
yearly_ml = assembler.transform(yearly_train).select(
    "features",
    F.col("strong_count_year").cast("double").alias("label"),
    "year",
    "year_index"
)

train_df, test_df = yearly_ml.randomSplit([0.8, 0.2], seed=42)
train_count = train_df.count()
test_count = test_df.count()

print(f"Train rows: {train_count}")
print(f"Test rows: {test_count}")

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.0,
    elasticNetParam=0.0
)

lr_model = lr.fit(train_df)

test_predictions = lr_model.transform(test_df)     .withColumn("residual", F.col("prediction") - F.col("label"))     .withColumn("abs_error", F.abs(F.col("prediction") - F.col("label")))

rmse_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
mae_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
r2_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse_value = rmse_eval.evaluate(test_predictions)
mae_value = mae_eval.evaluate(test_predictions)
r2_value = r2_eval.evaluate(test_predictions)

print(f"Test RMSE: {rmse_value}")
print(f"Test MAE: {mae_value}")
print(f"Test R2: {r2_value}")
print(f"Intercept: {lr_model.intercept}")
print(f"Coefficient(year_index): {lr_model.coefficients[0]}")

print("=== TEST PREDICTIONS ===")
test_predictions.select(
    "year",
    F.round("label", 3).alias("actual_strong_count"),
    F.round("prediction", 3).alias("predicted_strong_count"),
    F.round("residual", 3).alias("residual"),
    F.round("abs_error", 3).alias("abs_error")
).orderBy("year").show(10, truncate=False)

residual_stats = test_predictions.agg(
    F.count("*").alias("test_rows"),
    F.avg("residual").alias("mean_residual"),
    F.stddev("residual").alias("std_residual"),
    F.min("residual").alias("min_residual"),
    F.max("residual").alias("max_residual"),
    F.avg("abs_error").alias("mean_abs_error"),
    F.max("abs_error").alias("max_abs_error")
)

print("=== RESIDUAL STATS ===")
residual_stats.show(truncate=False)

metrics_df = spark.createDataFrame([
    (
        train_count,
        test_count,
        float(rmse_value),
        float(mae_value),
        float(r2_value),
        float(lr_model.intercept),
        float(lr_model.coefficients[0])
    )
], [
    "train_count",
    "test_count",
    "rmse",
    "mae",
    "r2",
    "intercept",
    "coef_year_index"
])

future_2026 = spark.createDataFrame([(2026, 2026 - 1900)], ["year", "year_index"])
future_2026_feat = assembler.transform(future_2026)

forecast_2026_total = lr_model.transform(future_2026_feat).select(
    "year",
    F.col("prediction").alias("predicted_strong_count_2026")
)

forecast_2026_total.show()

geo_points.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/geo_points")
geo_grid.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/geo_grid")
cat_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/catastrophic_events")
cat_intervals.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/catastrophic_intervals")
cat_stats.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/catastrophic_stats")
forecast_window.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/catastrophic_forecast_window")
yearly_strong.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/yearly_strong")
forecast_2027_total.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/forecast_2027_total")
geo_grid20.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/geo_grid20")


test_predictions.select(
    "year",
    F.round("label", 6).alias("actual_strong_count"),
    F.round("prediction", 6).alias("predicted_strong_count"),
    F.round("residual", 6).alias("residual"),
    F.round("abs_error", 6).alias("abs_error")
).orderBy("year").coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/model_test_predictions")

metrics_df.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/model_metrics")
residual_stats.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/model_residual_stats")
train_df.select("year", "label").orderBy("year").coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/model_train_data")
test_df.select("year", "label").orderBy("year").coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/model_test_data")

