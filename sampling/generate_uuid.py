from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import sys

def main():
    spark = SparkSession.builder.appName("uuid Generate").getOrCreate()
    datasets = sys.argv[1]
    documents_df = spark.read.json("zyf/DecorateLLM/rt/batch_results/" + datasets)
    documents_df = documents_df.withColumn("uuid", expr("uuid()"))
    documents_df.write.json("zyf/DecorateLLM/rt/data_with_uuid/" + datasets, mode = "overwrite")
    spark.stop()

if __name__ == '__main__':
    main()