#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession 
from pyspark.sql.functions import (
    col, to_timestamp, year as year_func, count, avg, max as spark_max,
    when, lag, lit, pow
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os
import shutil
for p in [output_yearly, output_val, output_test, output_forecast]:
    try:
        shutil.rmtree(p)
    except:
        pass
file_path = "/user/student/earth.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)
df = df.withColumn("time_ts", to_timestamp(col("time")))        .withColumn("year_raw", col("year").cast(IntegerType()))        .withColumn("month", col("month").cast(IntegerType()))        .withColumn("day_of_year", col("day_of_year").cast(IntegerType()))        .withColumn("hour", col("hour").cast(IntegerType()))        .withColumn("latitude", col("latitude").cast(DoubleType()))        .withColumn("longitude", col("longitude").cast(DoubleType()))        .withColumn("depth", col("depth").cast(DoubleType()))        .withColumn("mag", col("mag").cast(DoubleType()))

df = df.withColumn(
    "year",
    when(col("year_raw").isNull(), year_func(col("time_ts"))).otherwise(col("year_raw"))
).drop("year_raw")

df = df.dropna(subset=["year", "mag", "depth"])
df_train_source = df.filter(col("year") <= 2025)
yearly = df_train_source.groupBy("year").agg(
    count("*").alias("earthquake_count"),
    avg("mag").alias("avg_mag"),
    spark_max("mag").alias("max_mag"),
    avg("depth").alias("avg_depth"),
    count(when(col("mag") >= 4.5, True)).alias("strong_quakes_45plus"),
    count(when(col("mag") >= 5.0, True)).alias("strong_quakes_5plus"),
    count(when(col("mag") >= 6.0, True)).alias("strong_quakes_6plus")
).orderBy("year")
w = Window.orderBy("year")
yearly_features = yearly     .withColumn("lag_1", lag("earthquake_count", 1).over(w))     .withColumn("lag_2", lag("earthquake_count", 2).over(w))     .withColumn("lag_3", lag("earthquake_count", 3).over(w))     .withColumn("lag_4", lag("earthquake_count", 4).over(w))     .withColumn("lag_5", lag("earthquake_count", 5).over(w))     .withColumn("lag_6", lag("earthquake_count", 6).over(w))     .withColumn("mag_lag_1", lag("avg_mag", 1).over(w))     .withColumn("mag_lag_2", lag("avg_mag", 2).over(w))     .withColumn("depth_lag_1", lag("avg_depth", 1).over(w))     .withColumn("depth_lag_2", lag("avg_depth", 2).over(w))     .withColumn("strong45_lag_1", lag("strong_quakes_45plus", 1).over(w))     .withColumn("strong5_lag_1", lag("strong_quakes_5plus", 1).over(w))     .withColumn("strong6_lag_1", lag("strong_quakes_6plus", 1).over(w))     .withColumn("rolling_mean_3", avg("earthquake_count").over(w.rowsBetween(-2, 0)))     .withColumn("rolling_mean_5", avg("earthquake_count").over(w.rowsBetween(-4, 0)))     .withColumn("rolling_mean_7", avg("earthquake_count").over(w.rowsBetween(-6, 0)))     .withColumn("rolling_mag_3", avg("avg_mag").over(w.rowsBetween(-2, 0)))     .withColumn("rolling_depth_3", avg("avg_depth").over(w.rowsBetween(-2, 0)))     .withColumn("diff_1", col("earthquake_count") - col("lag_1"))     .withColumn("diff_2", col("lag_1") - col("lag_2"))     .withColumn("trend", col("year") - lit(1900))     .withColumn("trend_sq", pow(col("year") - lit(1900), lit(2)))

yearly_features = yearly_features.dropna().orderBy("year")

feature_cols = [
    "year",
    "trend",
    "trend_sq",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "rolling_mean_3", "rolling_mean_5", "rolling_mean_7",
    "diff_1", "diff_2",
    "mag_lag_1", "mag_lag_2",
    "depth_lag_1", "depth_lag_2",
    "rolling_mag_3", "rolling_depth_3",
    "strong45_lag_1", "strong5_lag_1", "strong6_lag_1"
]

label_col = "earthquake_count"
train = yearly_features.filter(col("year") <= 2015)
val = yearly_features.filter((col("year") >= 2016) & (col("year") <= 2019))
test = yearly_features.filter((col("year") >= 2020) & (col("year") <= 2025))
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_assembled = assembler.transform(train)
val_assembled = assembler.transform(val)
test_assembled = assembler.transform(test)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(train_assembled)
train_scaled = scaler_model.transform(train_assembled)
val_scaled = scaler_model.transform(val_assembled)
test_scaled = scaler_model.transform(test_assembled)
rmse_eval = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
mae_eval = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
r2_eval = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

models_info = []

# lINEAR
lr = LinearRegression(featuresCol="scaled_features", labelCol=label_col, regParam=0.1, elasticNetParam=0.2)
lr_model = lr.fit(train_scaled)
lr_val_pred = lr_model.transform(val_scaled)
models_info.append({
    "name": "LinearRegression",
    "model": lr_model,
    "val_pred": lr_val_pred,
    "rmse": rmse_eval.evaluate(lr_val_pred),
    "mae": mae_eval.evaluate(lr_val_pred),
    "r2": r2_eval.evaluate(lr_val_pred),
    "features_col": "scaled_features"
})

# Random Forest
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col,
    numTrees=300,
    maxDepth=8,
    minInstancesPerNode=1,
    seed=42
)
rf_model = rf.fit(train_assembled)
rf_val_pred = rf_model.transform(val_assembled)
models_info.append({
    "name": "RandomForest",
    "model": rf_model,
    "val_pred": rf_val_pred,
    "rmse": rmse_eval.evaluate(rf_val_pred),
    "mae": mae_eval.evaluate(rf_val_pred),
    "r2": r2_eval.evaluate(rf_val_pred),
    "features_col": "features"
})
#GBT Regressor
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=label_col,
    maxIter=120,
    maxDepth=5,
    maxBins=32,
    stepSize=0.05,
    seed=42
)
gbt_model = gbt.fit(train_assembled)
gbt_val_pred = gbt_model.transform(val_assembled)
models_info.append({
    "name": "GBTRegressor",
    "model": gbt_model,
    "val_pred": gbt_val_pred,
    "rmse": rmse_eval.evaluate(gbt_val_pred),
    "mae": mae_eval.evaluate(gbt_val_pred),
    "r2": r2_eval.evaluate(gbt_val_pred),
    "features_col": "features"
})
models_info = sorted(models_info, key=lambda x: x["rmse"])
best = models_info[0]

train_val = yearly_features.filter(col("year") <= 2019)
train_val_assembled = assembler.transform(train_val)
train_val_scaled = scaler_model.transform(train_val_assembled)
if best["name"] == "LinearRegression":
    final_model = LinearRegression(
        featuresCol="scaled_features",
        labelCol=label_col,
        regParam=0.1,
        elasticNetParam=0.2
    ).fit(train_val_scaled)
    test_input = test_scaled
    final_features_col = "scaled_features"

elif best["name"] == "RandomForest":
    final_model = RandomForestRegressor(
        featuresCol="features",
        labelCol=label_col,
        numTrees=300,
        maxDepth=8,
        minInstancesPerNode=1,
        seed=42
    ).fit(train_val_assembled)
    test_input = test_assembled
    final_features_col = "features"

else:
    final_model = GBTRegressor(
        featuresCol="features",
        labelCol=label_col,
        maxIter=120,
        maxDepth=5,
        maxBins=32,
        stepSize=0.05,
        seed=42
    ).fit(train_val_assembled)
    test_input = test_assembled
    final_features_col = "features"

test_pred = final_model.transform(test_input)
test_rmse = rmse_eval.evaluate(test_pred)
test_mae = mae_eval.evaluate(test_pred)
test_r2 = r2_eval.evaluate(test_pred)
print("\n===== Test results =====")
print("RMSE =", round(test_rmse, 4))
print("MAE  =", round(test_mae, 4))
print("R2   =", round(test_r2, 4))
test_pred.select("year", "earthquake_count", "prediction").orderBy("year").show(50, False)
yearly_full = yearly.orderBy("year").collect()

series = []
for row in yearly_full:
    series.append({
        "year": int(row["year"]),
        "earthquake_count": float(row["earthquake_count"]),
        "avg_mag": float(row["avg_mag"]) if row["avg_mag"] is not None else 0.0,
        "avg_depth": float(row["avg_depth"]) if row["avg_depth"] is not None else 0.0,
        "strong_45plus": float(row["strong_quakes_45plus"]) if row["strong_quakes_45plus"] is not None else 0.0,
        "strong_5plus": float(row["strong_quakes_5plus"]) if row["strong_quakes_5plus"] is not None else 0.0,
        "strong_6plus": float(row["strong_quakes_6plus"]) if row["strong_quakes_6plus"] is not None else 0.0
    })

def build_future_feature_row(future_year, series):
    prev1 = series[-1]
    prev2 = series[-2]
    prev3 = series[-3]
    prev4 = series[-4]
    prev5 = series[-5]
    prev6 = series[-6]
    lag_1 = prev1["earthquake_count"]
    lag_2 = prev2["earthquake_count"]
    lag_3 = prev3["earthquake_count"]
    lag_4 = prev4["earthquake_count"]
    lag_5 = prev5["earthquake_count"]
    lag_6 = prev6["earthquake_count"]
    rolling_mean_3 = (prev1["earthquake_count"] + prev2["earthquake_count"] + prev3["earthquake_count"]) / 3.0
    rolling_mean_5 = (
        prev1["earthquake_count"] + prev2["earthquake_count"] + prev3["earthquake_count"] +
        prev4["earthquake_count"] + prev5["earthquake_count"]
    ) / 5.0
    rolling_mean_7 = (
        prev1["earthquake_count"] + prev2["earthquake_count"] + prev3["earthquake_count"] +
        prev4["earthquake_count"] + prev5["earthquake_count"] + prev6["earthquake_count"] +
        prev6["earthquake_count"]
    ) / 7.0
    diff_1 = prev1["earthquake_count"] - prev2["earthquake_count"]
    diff_2 = prev2["earthquake_count"] - prev3["earthquake_count"]
    mag_lag_1 = prev1["avg_mag"]
    mag_lag_2 = prev2["avg_mag"]
    depth_lag_1 = prev1["avg_depth"]
    depth_lag_2 = prev2["avg_depth"]
    rolling_mag_3 = (prev1["avg_mag"] + prev2["avg_mag"] + prev3["avg_mag"]) / 3.0
    rolling_depth_3 = (prev1["avg_depth"] + prev2["avg_depth"] + prev3["avg_depth"]) / 3.0
    strong45_lag_1 = prev1["strong_45plus"]
    strong5_lag_1 = prev1["strong_5plus"]
    strong6_lag_1 = prev1["strong_6plus"]
    trend = future_year - 1900
    trend_sq = float(trend ** 2)
    return (
        future_year,
        float(trend),
        float(trend_sq),
        float(lag_1), float(lag_2), float(lag_3), float(lag_4), float(lag_5), float(lag_6),
        float(rolling_mean_3), float(rolling_mean_5), float(rolling_mean_7),
        float(diff_1), float(diff_2),
        float(mag_lag_1), float(mag_lag_2),
        float(depth_lag_1), float(depth_lag_2),
        float(rolling_mag_3), float(rolling_depth_3),
        float(strong45_lag_1), float(strong5_lag_1), float(strong6_lag_1)
    )

future_years = [2026, 2027, 2028, 2029, 2030]
future_predictions = []

for future_year in future_years:
    feature_row = build_future_feature_row(future_year, series)
    future_input = spark.createDataFrame([feature_row], schema=feature_cols)
    future_assembled = assembler.transform(future_input)
    if final_features_col == "scaled_features":
        future_ready = scaler_model.transform(future_assembled)
    else:
        future_ready = future_assembled

    pred_value = final_model.transform(future_ready).collect()[0]["prediction"]

    if pred_value < 0:
        pred_value = 0.0
    pred_value = float(pred_value)
    future_predictions.append((future_year, pred_value))
    series.append({
        "year": future_year,
        "earthquake_count": pred_value,
        "avg_mag": series[-1]["avg_mag"],
        "avg_depth": series[-1]["avg_depth"],
        "strong_45plus": series[-1]["strong_45plus"],
        "strong_5plus": series[-1]["strong_5plus"],
        "strong_6plus": series[-1]["strong_6plus"]
    })

forecast_df = spark.createDataFrame(future_predictions, ["year", "predicted_earthquake_count"])
forecast_df.show(50, False)
yearly.coalesce(1).write.mode("overwrite").option("header", True).csv("/user/student/Samsung_project_output/output_yearly")

best_val_pred = best["val_pred"].select("year", "earthquake_count", "prediction").orderBy("year")
best_val_pred.coalesce(1).write.mode("overwrite").option("header", True).csv("/user/student/Samsung_project_output/output_val")
test_pred.select("year", "earthquake_count", "prediction")     .orderBy("year")     .coalesce(1).write.mode("overwrite").option("header", True).csv("/user/student/Samsung_project_output/output_test")
forecast_df.coalesce(1).write.mode("overwrite").option("header", True).csv("/user/student/Samsung_project_output/output_forecast")

