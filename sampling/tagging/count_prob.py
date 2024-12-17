from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, lit, when, abs, sqrt
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import expr
import sys


def calculate_probabilities(df, alpha, beta, gamma):
    # count num of each level label
    count_a = df.groupBy("first_level_tag").count().withColumnRenamed("count", "count_a") # 所属一级标签有多少个
    count_b = df.groupBy("first_level_tag", "second_level_tag").count().withColumnRenamed("count", "count_b") # 所属二级标签有多少个
    count_c = df.groupBy("first_level_tag", "second_level_tag", "third_level_tag").count().withColumnRenamed("count", "count_c") # 所属三级标签有多少个

    # calculate the total weight of each first level label
    sum_a = count_a.withColumn("weight_a", F.pow(F.col("count_a"), alpha)) \
                    .groupBy().sum("weight_a").withColumnRenamed("sum(weight_a)", "sum_weight_a")

    # calculate the total weight of each second level label
    sum_b = count_b.withColumn("weight_b", F.pow(F.col("count_b"), beta)) \
                    .groupBy("first_level_tag").sum("weight_b").withColumnRenamed("sum(weight_b)", "sum_weight_b")

    # calculate the total weight of each third level label
    sum_c = count_c.withColumn("weight_c", F.pow(F.col("count_c"), gamma)) \
                    .groupBy("first_level_tag", "second_level_tag").sum("weight_c").withColumnRenamed("sum(weight_c)", "sum_weight_c")

    # join
    df = df.join(count_a, "first_level_tag") \
            .join(count_b, ["first_level_tag", "second_level_tag"]) \
            .join(sum_b, "first_level_tag") \
            .join(count_c, ["first_level_tag", "second_level_tag", "third_level_tag"]) \
            .join(sum_c, ["first_level_tag", "second_level_tag"])


    # calculate prob
    df = df.withColumn("probability",
                       (F.pow(F.col("count_a"), alpha)) *
                       (F.pow(F.col("count_b"), beta) / F.col("sum_weight_b")) *
                       (F.pow(F.col("count_c"), gamma) / F.col("sum_weight_c")) /
                       F.col("count_c"))

    return df

if __name__ == '__main__':
    datasets = sys.argv[1]
    # from json to dataframe
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    df = spark.read.json("zyf/DecorateLLM/rt/data_with_uuid/" + datasets).filter(col("in_tree") == True)
    alpha = beta = gamma = 0.5
    df_with_probabilities = calculate_probabilities(df, alpha, beta, gamma)
    final_df = df_with_probabilities.select("uuid", "first_level_tag", "second_level_tag", "third_level_tag", "probability")
    final_df.repartition(1).write.json("zyf/DecorateLLM/rt/tagging_probability_new/" + datasets, mode="overwrite")
    spark.stop()