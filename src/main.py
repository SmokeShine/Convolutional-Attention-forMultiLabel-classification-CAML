import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.feature import Word2Vec
from pyspark.sql.types import StructType,IntegerType,StringType,TimestampType

spark = SparkSession.builder.appName('BigDataHealthSparkApp').getOrCreate()

note_event_data_location = "hdfs://localhost:9000/input/raw_data/NOTEEVENTS.csv"
diagnoses_icd_data_location = "hdfs://localhost:9000/input/raw_data/DIAGNOSES_ICD.csv"

def data_processing():
    
    note_events_df = spark.read.format("csv").\
        option("header",True)\
        .option("multiline", "true")\
        .load(note_event_data_location)    
    
    diagnoses_icd_df = spark.read.format("csv").\
        option("header",True).\
        load(diagnoses_icd_data_location)
    
    note_events_df.printSchema()
    note_events_df.show(2)
    diagnoses_icd_df.printSchema()
    diagnoses_icd_df.show(2)

if __name__ == '__main__':
    data_processing()


