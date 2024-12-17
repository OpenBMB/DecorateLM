from pyspark.sql import SparkSession
from pyspark.sql.functions import col
if __name__ == '__main__':
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    # uuid path
    sampled_ids_df = spark.read.json("/user/tc_agi/zyf/DecorateLLM/tagging_uuid/baike_chinese_new_all_uuids.jsonl")
    # sample
    df = spark.read.json("zyf/DecorateLLM/rt/tagging_data_with_uuid/baike_chinese_new_all")
    df = df.withColumnRenamed("uuid", "df_uuid")
    result_df = sampled_ids_df.join(df, sampled_ids_df.uuid == df.df_uuid, "left")
    result_df.show()
    result_df.drop("df_uuid").write.json("zyf/DecorateLLM/rt/rating_sample_results/baike_chinese_new_all", mode = 'overwrite')
    spark.stop()