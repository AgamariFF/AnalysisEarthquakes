#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, trim, lower, count, lit, when
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer, IndexToString
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

file_path = "/user/student/earth.csv"

df = spark.read     .option("header", True)     .option("inferSchema", True)     .csv(file_path)

expected_cols = [
    "id", "time", "year", "month", "day_of_year", "hour",
    "latitude", "longitude", "depth", "mag", "magType", "place", "type",
    "tsunami", "mag_category", "depth_category", "nst", "gap", "dmin",
    "rms", "net", "status"
]

existing_cols = [c for c in expected_cols if c in df.columns]
df = df.select(*existing_cols)

numeric_cols = [
    "year", "month", "day_of_year", "hour",
    "latitude", "longitude", "depth", "mag",
    "nst", "gap", "dmin", "rms"
]

for c in numeric_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast(DoubleType()))

text_cols = ["magType", "type", "mag_category", "depth_category", "net", "status"]
for c in text_cols:
    if c in df.columns:
        df = df.withColumn(c, trim(col(c).cast(StringType())))

df = df.withColumn("depth_category_norm", lower(trim(col("depth_category"))))

valid_depths = [
    "shallow (0-70km)",
    "intermediate (70-300km)",
    "deep (300-700km)"
]

df = df.filter(col("depth_category_norm").isin(valid_depths))
df = df.dropna(subset=["year", "depth_category_norm", "latitude", "longitude", "depth", "mag"])

print("Depth category distribution:")
df.groupBy("depth_category_norm").count().show(truncate=False)

train_df_full = df.filter(col("year") <= 2022)
test_df = df.filter(col("year") > 2022)

print("Train years:")
train_df_full.selectExpr("min(year) as min_year", "max(year) as max_year").show()

print("Test years:")
test_df.selectExpr("min(year) as min_year", "max(year) as max_year").show()

train_df = train_df_full.sample(withReplacement=False, fraction=0.25, seed=42)

label_indexer = StringIndexer(
    inputCol="depth_category_norm",
    outputCol="label",
    handleInvalid="error"
).fit(train_df)

train_df = label_indexer.transform(train_df)
test_df = label_indexer.transform(test_df)

labels = label_indexer.labels
print("Labels order:", labels)

class_counts = train_df.groupBy("label").count()
total_count = train_df.count()
num_classes = class_counts.count()

class_weights = class_counts.withColumn(
    "classWeight",
    F.lit(float(total_count)) / (F.col("count") * F.lit(float(num_classes)))
).select("label", "classWeight")

train_df = train_df.join(class_weights, on="label", how="left")
test_df = test_df.join(class_weights, on="label", how="left")

print("Class weights:")
class_weights.orderBy("label").show(truncate=False)

categorical_cols = [c for c in ["magType", "type", "mag_category", "net", "status"] if c in df.columns]

numeric_feature_cols = [c for c in [
    "month", "day_of_year", "hour",
    "latitude", "longitude", "depth", "mag",
    "nst", "gap", "dmin", "rms"
] if c in df.columns]

stages = []

if numeric_feature_cols:
    imputer = Imputer(
        inputCols=numeric_feature_cols,
        outputCols=[f"{c}_imp" for c in numeric_feature_cols]
    ).setStrategy("median")
    stages.append(imputer)
    numeric_out = [f"{c}_imp" for c in numeric_feature_cols]
else:
    numeric_out = []

encoded_cols = []
for c in categorical_cols:
    idx_col = f"{c}_idx"
    ohe_col = f"{c}_ohe"

    stages.append(StringIndexer(
        inputCol=c,
        outputCol=idx_col,
        handleInvalid="keep"
    ))
    stages.append(OneHotEncoder(
        inputCols=[idx_col],
        outputCols=[ohe_col],
        handleInvalid="keep"
    ))
    encoded_cols.append(ohe_col)

assembler = VectorAssembler(
    inputCols=numeric_out + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)
stages.append(assembler)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    family="multinomial",
    maxIter=80,
    regParam=0.02,
    elasticNetParam=0.0,
    standardization=True,
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction"
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=80,
    maxDepth=8,
    maxBins=32,
    minInstancesPerNode=10,
    featureSubsetStrategy="sqrt",
    seed=42,
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction"
)

lr_pipeline = Pipeline(stages=stages + [lr])
rf_pipeline = Pipeline(stages=stages + [rf])

lr_model = lr_pipeline.fit(train_df)
rf_model = rf_pipeline.fit(train_df)

lr_pred = lr_model.transform(test_df)
rf_pred = rf_model.transform(test_df)

acc_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedFMeasure"
)

def metric_by_label(pred_df, metric_name, label_idx):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric_name,
        metricLabel=float(label_idx)
    )
    return evaluator.evaluate(pred_df)

def per_class_metrics(pred_df, labels, model_name):
    rows = []
    for i, lbl in enumerate(labels):
        precision_val = metric_by_label(pred_df, "precisionByLabel", i)
        recall_val = metric_by_label(pred_df, "recallByLabel", i)
        f1_val = metric_by_label(pred_df, "fMeasureByLabel", i)
        rows.append((model_name, lbl, float(precision_val), float(recall_val), float(f1_val)))
    return spark.createDataFrame(rows, ["model_name", "class_name", "precision", "recall", "f1"])

def evaluate_model(pred_df, model_name, labels):
    acc = acc_eval.evaluate(pred_df)
    wf1 = f1_eval.evaluate(pred_df)

    class_df = per_class_metrics(pred_df, labels, model_name)

    intermediate_row = class_df.filter(
        col("class_name") == "intermediate (70-300km)"
    ).collect()

    intermediate_f1 = intermediate_row[0]["f1"] if intermediate_row else 0.0

    print(f"{model_name} accuracy: {acc}")
    print(f"{model_name} weighted F1: {wf1}")
    print(f"{model_name} intermediate F1: {intermediate_f1}")

    return {
        "accuracy": float(acc),
        "weighted_f1": float(wf1),
        "intermediate_f1": float(intermediate_f1),
        "class_df": class_df
    }

lr_metrics = evaluate_model(lr_pred, "LogisticRegression", labels)
rf_metrics = evaluate_model(rf_pred, "RandomForestClassifier", labels)

if (rf_metrics["intermediate_f1"] > lr_metrics["intermediate_f1"]) or (
    rf_metrics["intermediate_f1"] == lr_metrics["intermediate_f1"] and
    rf_metrics["weighted_f1"] >= lr_metrics["weighted_f1"]
):
    best_model = rf_model
    best_pred = rf_pred
    best_name = "RandomForestClassifier"
    best_metrics = rf_metrics
else:
    best_model = lr_model
    best_pred = lr_pred
    best_name = "LogisticRegression"
    best_metrics = lr_metrics

print("Best model:", best_name)

label_converter = IndexToString(
    inputCol="prediction",
    outputCol="predicted_depth_category",
    labels=labels
)

best_pred_named = label_converter.transform(best_pred)

label_map = F.create_map(*[
    x for i, lbl in enumerate(labels)
    for x in (F.lit(float(i)), F.lit(lbl))
])

best_pred_result = best_pred_named.withColumn(
    "true_depth_category",
    label_map[col("label")]
).withColumn(
    "is_correct",
    when(col("true_depth_category") == col("predicted_depth_category"), lit(1)).otherwise(lit(0))
)

best_pred_result = best_pred_result.withColumn(
    "probability_array",
    vector_to_array(col("probability"))
)

best_pred_export = best_pred_result.select(
    "year", "month", "day_of_year", "hour",
    "latitude", "longitude", "depth", "mag",
    "depth_category_norm",
    "true_depth_category",
    "predicted_depth_category",
    F.round(col("prediction"), 6).alias("prediction"),
    F.round(col("label"), 6).alias("label"),
    "is_correct",
    F.round(col("probability_array")[0], 6).alias("prob_shallow"),
    F.round(col("probability_array")[1], 6).alias("prob_intermediate"),
    F.round(col("probability_array")[2], 6).alias("prob_deep")
)

print("===== TEST SAMPLE =====")
best_pred_export.show(30, truncate=False)

quality_summary_df = spark.createDataFrame([
    (
        best_name,
        float(best_metrics["accuracy"]),
        float(best_metrics["weighted_f1"]),
        float(best_metrics["intermediate_f1"])
    )
], ["model_name", "accuracy", "weighted_f1", "intermediate_f1"])

print("===== QUALITY SUMMARY =====")
quality_summary_df.show(truncate=False)
confusion_matrix_df = best_pred_result.groupBy("true_depth_category", "predicted_depth_category")     .agg(count("*").alias("events_count"))     .orderBy("true_depth_category", "predicted_depth_category")

print("===== CONFUSION MATRIX =====")
confusion_matrix_df.show(10, truncate=False)

year_cat_counts = df.groupBy("year", "depth_category_norm").agg(
    count("*").alias("events_count")
)

year_totals = df.groupBy("year").agg(
    count("*").alias("total_events")
)

year_cat_share = year_cat_counts.join(year_totals, on="year", how="inner")     .withColumn("share", col("events_count") / col("total_events"))     .orderBy("year", "depth_category_norm")

forecast_total_df = spark.createDataFrame(
    [
        (2026, 29937.88181578379),
        (2027, 31148.5507830383),
        (2028, 31630.87176990843),
        (2029, 31938.55804597806),
        (2030, 32082.273980635073)
    ],
    ["year", "predicted_earthquake_count"]
)

recent_base = df.filter(col("year") >= 2024)
if recent_base.count() == 0:
    recent_base = df.filter(col("year") >= 2020)

sample_limit = 50000
feature_cols = [c for c in df.columns if c != "depth_category_norm"]
base_sample = recent_base.select(*feature_cols).limit(sample_limit)

forecast_rows = []

for row in forecast_total_df.collect():
    target_year = int(row["year"])
    target_total = float(row["predicted_earthquake_count"])

    base_year = base_sample.withColumn("year", lit(float(target_year)))
    forecast_depth_pred = best_model.transform(base_year)
    forecast_depth_pred = label_converter.transform(forecast_depth_pred)

    predicted_distribution = forecast_depth_pred.groupBy("predicted_depth_category").agg(
        count("*").alias("sample_predicted_count")
    )

    all_classes_df = spark.createDataFrame([
        ("shallow (0-70km)",),
        ("intermediate (70-300km)",),
        ("deep (300-700km)",)
    ], ["predicted_depth_category"])

    predicted_distribution = all_classes_df.join(
        predicted_distribution,
        on="predicted_depth_category",
        how="left"
    ).fillna({"sample_predicted_count": 0})

    sample_total = forecast_depth_pred.count()

    predicted_distribution = predicted_distribution.withColumn(
        "predicted_share",
        F.when(F.lit(sample_total) > 0,
               F.col("sample_predicted_count") / F.lit(sample_total)).otherwise(F.lit(0.0))
    ).withColumn(
        "forecast_events",
        F.col("predicted_share") * F.lit(target_total)
    ).withColumn(
        "year",
        F.lit(target_year)
    )

    forecast_rows.append(predicted_distribution)

forecast_depth_forecast = forecast_rows[0]
for df_part in forecast_rows[1:]:
    forecast_depth_forecast = forecast_depth_forecast.unionByName(df_part)

forecast_depth_forecast = forecast_depth_forecast.orderBy("year", "predicted_depth_category")

print("===== FORECAST DEPTH CATEGORY =====")
forecast_depth_forecast.show(truncate=False)

base_out = "/user/student/Samsung_project_output"
best_pred_export.coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"{base_out}/depth_category_test_predictions_final"
)

best_metrics["class_df"].coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"{base_out}/depth_category_per_class_metrics_final"
)

year_cat_share.coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"{base_out}/depth_category_yearly_share_final"
)

forecast_depth_forecast.coalesce(1).write.mode("overwrite").option("header", True).csv(
    f"{base_out}/depth_category_forecast_2026_2030_final"
)


# In[ ]:




