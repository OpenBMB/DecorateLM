from pyspark.sql import SparkSession
from pyspark.sql.functions import col, exp, sum as _sum, monotonically_increasing_id, udf, lag, lit, mean, stddev
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window
import random
import sys

def check_validation(score):
    try:
        if float(score) > 100 or float(score) < 0:
            return None
        return float(score)
    except ValueError:
        return None 

def main():
    spark = SparkSession.builder.appName("Document Sampling").getOrCreate()
    path = "zyf/DecorateLLM/rt/data_with_uuid/*/*.json"
    documents_df = spark.read.json(path)
    fields = ["educational_value", "expertise", "fact&trivia", "obscurity", "reasoning_level", "story-like", "structural_format", "subjectivity"]
    check_validation_udf = udf(check_validation, FloatType())
    for field in fields:
        documents_df = documents_df.withColumn(field, check_validation_udf(col(field))).filter(col(field).isNotNull())
    stats = {field: {'mean': 0, 'std': 1} for field in fields} 
    for field in fields:
        stats[field]['mean'] = documents_df.select(mean(col(field))).collect()[0][0]
        stats[field]['std'] = documents_df.select(stddev(col(field))).collect()[0][0]
    print(stats)
    for field in fields:
        documents_df = documents_df.withColumn(field, (col(field) - stats[field]['mean']) / stats[field]['std']).filter(col(field).isNotNull())
        documents_df = documents_df.withColumn(field, exp(col(field)))
        
    selected_columns = ["uuid"] + fields
    final_df = documents_df.select(*selected_columns)
    final_df.repartition(1).write.json("zyf/DecorateLLM/rt/rating_probability_5in1/", mode="overwrite")

    spark.stop()

if __name__ == '__main__':
    main()