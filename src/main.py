from typing import Pattern
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
# from pyspark.mllib.feature import Word2Vec
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, lit
from pyspark.sql import functions as F
from pyspark.ml.feature import *
from pyspark.ml.linalg import Vectors
import mymodels
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# Util function from assignment
from utils import train, evaluate
from torch.utils.data import TensorDataset, Dataset
import os
import pickle
from torch.utils.data import DataLoader
from ast import literal_eval
from scipy import sparse
import sys 
from plots import plot_learning_curves, plot_confusion_matrix

spark = SparkSession.builder.appName('BigDataHealthSparkApp').getOrCreate()

server = 'hdfs://localhost:9000'
note_event_data_location = f"{server}/input/raw_data/NOTEEVENTS.csv"
diagnoses_icd_data_location = f"{server}/input/raw_data/DIAGNOSES_ICD.csv"
procedures_icd_data_location = f"{server}/input/raw_data/PROCEDURES_ICD.csv"
word2vec_embedding_location = f"{server}/intermediate/word2vec_embedding"
vocab_dict_file="vocab_dict.pkl"
embedding_layer_file="embedding_layer.pkl"
reverse_vocab_dict_file="reverse_vocab_dict.pkl"
icd9_index_dict_file="icd9_dict_index.pkl"
reverse_icd9_index_dict_file="reverse_icd9_dict_index.pkl"
# save to hadoop if required
PATH_OUTPUT="./"

embedding_size = 100
BATCH_SIZE = 32
USE_CUDA = True
# RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
NUM_WORKERS = 3  # For debugging
NUM_EPOCHS = 10
device = torch.device(
    "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
if device.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_processing():

    note_events_df = spark.read.format("csv").\
        option("header", True)\
        .option("multiline", "true")\
        .option("quote", '"')\
        .option("escape", '"')\
        .load(note_event_data_location)

    # from pyspark.sql.functions import length
    # test=note_events_df.withColumn('x',length(note_events_df["TEXT"]))
    # import pdb;pdb.set_trace()

    diagnoses_icd_df = spark.read.format("csv").\
        option("header", True).\
        load(diagnoses_icd_data_location)

    procedures_icd_df = spark.read.format("csv").\
        option("header", True).\
        load(procedures_icd_data_location)
    
    # note_events_df.printSchema()
    # note_events_df.show(2)

    # (Pdb) note_events_df.filter(note_events_df["ROW_ID"]==175).select("TEXT").take(1)[0]
    # Row(TEXT='"Admission Date:  [**2118-6-2**]
    # Discharge Date:  [**2118-6-14**]\n\nDate of Birth:
    # Sex:  F\n\nService:  MICU and then to [**Doctor Last Name **] Medicine\n\nHISTORY OF PRESENT ILLNESS:
    # This is an 81-year-old female\nwith a history of emphysema (not on home O2),
    # who presents\nwith three days of shortness of breath thought by her primary\ncare doctor to be a COPD flare.
    # Two days prior to admission,\nshe was started on a prednisone taper and one day prior to\nadmission
    # she required oxygen at home in order to maintain\noxygen saturation greater than 90%.
    # She has also been on\nlevofloxacin and nebulizers, and was not getting better, and\npresented to the [**Hospital1 18**]
    # Emergency Room.\n\nIn the [**Hospital3 **] Emergency Room, her oxygen saturation was\n100% on CPAP.
    # She was not able to be weaned off of this\ndespite nebulizer treatment and Solu-Medrol 125 mg IV x2.\n\n
    # Review of systems is negative for the following:  Fevers,\nchills, nausea, vomiting, night sweats, change in weight,
    # \ngastrointestinal complaints, neurologic changes, rashes,\npalpitations, orthopnea.  Is positive for the following:
    # \nChest pressure occasionally with shortness of breath with\nexertion, some shortness of breath that is positionally\nrelated,
    # but is improved with nebulizer treatment.\n\nPAST MEDICAL HISTORY:\n1. COPD.  Last pulmonary function tests in [**2117-11-3**]
    # \ndemonstrated a FVC of 52% of predicted, a FEV1 of 54% of\npredicted, a MMF of 23% of predicted, and a FEV1:FVC ratio of\n67%
    # of predicted, that does not improve with bronchodilator\ntreatment.  The FVC, however, does significantly improve with\nbronchodilator
    # treatment consistent with her known reversible\nair flow obstruction in addition to an underlying restrictive\nventilatory defect.
    # The patient has never been on home\noxygen prior to this recent episode.  She has never been on\nsteroid taper or been intubated in
    # the past.\n2. Lacunar CVA.  MRI of the head in [**2114-11-4**]\ndemonstrates ""mild degree of multiple small foci of high T2')

    # root

    # |-- ROW_ID: string (nullable = true)
    # |-- SUBJECT_ID: string (nullable = true)
    # |-- HADM_ID: string (nullable = true)
    # |-- CHARTDATE: string (nullable = true)
    # |-- CHARTTIME: string (nullable = true)
    # |-- STORETIME: string (nullable = true)
    # |-- CATEGORY: string (nullable = true)
    # |-- DESCRIPTION: string (nullable = true)
    # |-- CGID: string (nullable = true)
    # |-- ISERROR: string (nullable = true)
    # |-- TEXT: string (nullable = true)

    # +------+----------+-------+----------+---------+---------+-----------------+-----------+----+-------+--------------------+
    # |ROW_ID|SUBJECT_ID|HADM_ID| CHARTDATE|CHARTTIME|STORETIME|         CATEGORY|DESCRIPTION|CGID|ISERROR|                TEXT|
    # +------+----------+-------+----------+---------+---------+-----------------+-----------+----+-------+--------------------+
    # |   174|     22532| 167853|2151-08-04|     null|     null|Discharge summary|     Report|null|   null|Admission Date:  ...|
    # |   175|     13702| 107527|2118-06-14|     null|     null|Discharge summary|     Report|null|   null|"Admission Date: ...|
    # +------+----------+-------+----------+---------+---------+-----------------+-----------+----+-------+--------------------+
    # only showing top 2 rows

    # diagnoses_icd_df.printSchema()
    # diagnoses_icd_df.show(2)

    # root
    # |-- ROW_ID: string (nullable = true)
    # |-- SUBJECT_ID: string (nullable = true)
    # |-- HADM_ID: string (nullable = true)
    # |-- SEQ_NUM: string (nullable = true)
    # |-- ICD9_CODE: string (nullable = true)

    # +------+----------+-------+-------+---------+
    # |ROW_ID|SUBJECT_ID|HADM_ID|SEQ_NUM|ICD9_CODE|
    # +------+----------+-------+-------+---------+
    # |  1297|       109| 172335|      1|    40301|
    # |  1298|       109| 172335|      2|      486|
    # +------+----------+-------+-------+---------+
    # only showing top 2 rows

    # procedures_icd_df.printSchema()
    # procedures_icd_df.show(2)
    # root
    # |-- ROW_ID: string (nullable = true)
    # |-- SUBJECT_ID: string (nullable = true)
    # |-- HADM_ID: string (nullable = true)
    # |-- SEQ_NUM: string (nullable = true)
    # |-- ICD9_CODE: string (nullable = true)

    # +------+----------+-------+-------+---------+
    # |ROW_ID|SUBJECT_ID|HADM_ID|SEQ_NUM|ICD9_CODE|
    # +------+----------+-------+-------+---------+
    # |   944|     62641| 154460|      3|     3404|
    # |   945|      2592| 130856|      1|     9671|
    # +------+----------+-------+-------+---------+

    # https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/HospitalQualityInits/Downloads/HospitalAppendix_F.pdf
    diagnoses_icd_df.registerTempTable("diagnoses_icd_df")
    diagnoses_icd_df = spark.sql(
        """
        Select *,
        case when 
            substring(ICD9_CODE,0,1)=='E' 
            then 
            concat(substring(ICD9_CODE,0,4),'.',substring(ICD9_CODE,5,length(ICD9_CODE)))
            else 
            concat(substring(ICD9_CODE,0,3),'.',substring(ICD9_CODE,4,length(ICD9_CODE)))
            end as cleaned_ICD
        from
        diagnoses_icd_df
        where 
        ICD9_CODE is not null
        """)
    # diagnoses_icd_df.show(1)
    # +------+----------+-------+-------+---------+-----------+
    # |ROW_ID|SUBJECT_ID|HADM_ID|SEQ_NUM|ICD9_CODE|cleaned_ICD|
    # +------+----------+-------+-------+---------+-----------+
    # |  1297|       109| 172335|      1|    40301|     403.01|
    # +------+----------+-------+-------+---------+-----------+

    # +------+----------+-------+-------+---------+-----------+
    # |ROW_ID|SUBJECT_ID|HADM_ID|SEQ_NUM|ICD9_CODE|cleaned_ICD|
    # +------+----------+-------+-------+---------+-----------+
    # |  1519|       115| 114585|     18|    E9320|     E932.0|
    # +------+----------+-------+-------+---------+-----------+

    # (Pdb) print(diagnoses_icd_df.select("cleaned_ICD").distinct().show())
    # +-----------+
    # |cleaned_ICD|
    # +-----------+
    # |     458.29|
    # |      191.9|
    # |     415.11|
    # |      286.9|
    # |     928.11|
    # |     997.62|
    # |     807.05|
    # |      151.0|
    # |     810.03|
    # |     E937.8|
    # |      958.3|
    # |      886.0|
    # |      536.8|
    # |      V17.1|
    # |      110.9|
    # |     803.00|
    # |     647.84|
    # |      005.1|
    # |     787.20|
    # |      850.9|
    # +-----------+
    # only showing top 20 rows

    procedures_icd_df.registerTempTable("procedures_icd_df")
    procedures_icd_df = spark.sql(
        """
        Select *,
        concat(substring(ICD9_CODE,0,2),'.',substring(ICD9_CODE,3,length(ICD9_CODE)))
        as cleaned_ICD
        from
        procedures_icd_df
        where 
        ICD9_CODE is not null
        """)
    # procedures_icd_df.show(1)
    # +------+----------+-------+-------+---------+-----------+
    # |ROW_ID|SUBJECT_ID|HADM_ID|SEQ_NUM|ICD9_CODE|cleaned_ICD|
    # +------+----------+-------+-------+---------+-----------+
    # |   944|     62641| 154460|      3|     3404|      34.04|
    # +------+----------+-------+-------+---------+-----------+

    # (Pdb) print(procedures_icd_df.select("cleaned_ICD").distinct().show())
    # +-----------+
    # |cleaned_ICD|
    # +-----------+
    # |      46.10|
    # |      37.21|
    # |      78.59|
    # |      83.65|
    # |      79.14|
    # |      08.20|
    # |      42.87|
    # |      34.71|
    # |      00.71|
    # |      17.42|
    # |      36.12|
    # |      33.22|
    # |      04.42|
    # |      50.69|
    # |      84.21|
    # |       34.4|
    # |      77.73|
    # |      38.80|
    # |      45.51|
    # |      89.38|
    # +-----------+
    # only showing top 20 rows

    combined_diagnoses_procedures_df = diagnoses_icd_df.union(
        procedures_icd_df)

    # print(diagnoses_icd_df.count(),procedures_icd_df.count(),combined_diagnoses_procedures_df.count())
    # 651047 240095 891142
    # combined_diagnoses_procedures_df.show(1)
    # +------+----------+-------+-------+---------+-----------+
    # |ROW_ID|SUBJECT_ID|HADM_ID|SEQ_NUM|ICD9_CODE|cleaned_ICD|
    # +------+----------+-------+-------+---------+-----------+
    # |  1297|       109| 172335|      1|    40301|     403.01|
    # +------+----------+-------+-------+---------+-----------+

    # combined_diagnoses_procedures_df.write.mode("overwrite").parquet(f"{server}/intermediate/combined_diagnoses_procedures_df")

    #     (hw5) hdoop@pop-os:/home/ubuntu/Documents/CAML_GroupProject/src$ hdfs dfs -ls /intermediate/combined_diagnoses_procedures_df
    # Found 8 items
    # -rw-r--r--   3 hdoop supergroup          0 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/_SUCCESS
    # -rw-r--r--   3 hdoop supergroup    1549843 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00000-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet
    # -rw-r--r--   3 hdoop supergroup    1443770 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00001-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet
    # -rw-r--r--   3 hdoop supergroup    1341835 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00002-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet
    # -rw-r--r--   3 hdoop supergroup    1295351 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00003-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet
    # -rw-r--r--   3 hdoop supergroup     733833 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00004-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet
    # -rw-r--r--   3 hdoop supergroup    1878122 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00005-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet
    # -rw-r--r--   3 hdoop supergroup    1131905 2021-04-10 15:03 /intermediate/combined_diagnoses_procedures_df/part-00006-a85c5f31-c610-4126-b02a-f84e24a71093-c000.snappy.parquet

    # combined_diagnoses_procedures_df.select("cleaned_ICD").distinct().count()
    # 9017

    # import pdb;pdb.set_trace()
    note_events_df = note_events_df.orderBy(
        ["HADM_ID", "CHARTTIME"]).dropDuplicates(subset=['HADM_ID'])
    discharge_summary_df = note_events_df.filter(
        note_events_df["CATEGORY"] == "Discharge summary")
    # https://spark.apache.org/docs/latest/ml-features#tokenizer
    # https://stackoverflow.com/questions/6053541/regex-every-non-alphanumeric-character-except-white-space-or-colon
    # https://stackoverflow.com/questions/48278489/how-to-create-a-custom-tokenizer-in-pyspark-ml/48279714
    regexTokenizer = RegexTokenizer(
        inputCol="TEXT", outputCol="words", pattern=r'\w+', gaps=False)
    # |[admission, date, 2151, 7, 16, discharge, date, 2151, 8, 4, service, addendum,
    # radiologic, studies, radiologic, studies, also, included, a, chest, ct, which,
    # confirmed, cavitary, lesions, in, the, left, lung, apex, consistent, with, infectious,
    # process, tuberculosis, this, also, moderate, sized, left, pleural, effusion, head, ct, head, ct,
    # showed, no, intracranial, hemorrhage, or, mass, effect, but, old, infarction, consistent, with, past,
    # medical, history, abdominal, ct, abdominal, ct, showed, lesions, of, t10, and, sacrum, most, likely, secondary,
    # to, osteoporosis, these, can, be, followed, by, repeat, imaging, as, an, outpatient, first, name8, namepattern2,
    # first, name4, namepattern1, 1775, last, name, namepattern1, m, d, md, number, 1, 1776, dictated, by, hospital, 1807,
    # medquist36, d, 2151, 8, 5, 12, 11, t, 2151, 8, 5, 12, 21, job, job, number, 1808]|

    tokenized = regexTokenizer.transform(discharge_summary_df)

    # remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    # tokenized_stop_words_removal=remover.transform(tokenized)
    # cant do this.. converting "no infection" to "infection"

    # |[admission, date, 2151, 7, 16, discharge, date, 2151, 8, 4, service,
    # addendum, radiologic, studies, radiologic, studies, also, included, chest,
    # ct, confirmed, cavitary, lesions, left, lung, apex, consistent, infectious,
    # process, tuberculosis, also, moderate, sized, left, pleural, effusion, head,
    # ct, head, ct, showed, intracranial, hemorrhage, mass, effect, old, infarction,
    # consistent, past, medical, history, abdominal, ct, abdominal, ct, showed, lesions,
    # t10, sacrum, likely, secondary, osteoporosis, followed, repeat, imaging, outpatient,
    # first, name8, namepattern2, first, name4, namepattern1, 1775, last, name, namepattern1,
    # m, d, md, number, 1, 1776, dictated, hospital, 1807, medquist36,
    # d, 2151, 8, 5, 12, 11, 2151, 8, 5, 12, 21, job, job, number, 1808]|

    # https://stackoverflow.com/questions/53951215/how-do-i-remove-words-numerics-pyspark
    is_numeric_udf = udf(is_numeric, BooleanType())
    filter_length_udf = udf(
        lambda row: [x for x in row if not is_numeric(x)], ArrayType(StringType()))
    alphanumeric_tokens = tokenized.withColumn(
        'alphanumeric', filter_length_udf(col('words')))
    # |[admission, date, discharge, date, service, addendum, radiologic, studies, radiologic,
    # studies, also, included, chest, ct, confirmed, cavitary, lesions, left,
    # lung, apex, consistent, infectious, process, tuberculosis, also, moderate,
    # sized, left, pleural, effusion, head, ct, head, ct, showed, intracranial,
    # hemorrhage, mass, effect, old, infarction, consistent, past, medical, history,
    # abdominal, ct, abdominal, ct, showed, lesions, t10, sacrum, likely, secondary,
    # osteoporosis, followed, repeat, imaging, outpatient, first, name8, namepattern2,
    # first, name4, namepattern1, last, name, namepattern1, m, d, md, number, dictated,
    # hospital, medquist36, d, job, job, number]|

    # Attach label to this alphanumeric tokens
    # (Pdb) alphanumeric_tokens.columns
    # ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME',
    # 'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID', 'ISERROR',
    # 'TEXT', 'words', 'filtered', 'alphanumeric']

    # (Pdb) combined_diagnoses_procedures_df.columns
    # ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE', 'cleaned_ICD']

    # (Pdb) print(combined_diagnoses_procedures_df.select("HADM_ID").distinct().count())
    # 58929
    # (Pdb) print(alphanumeric_tokens.select("HADM_ID").distinct().count())
    # 52726

    combined_diagnoses_procedures_summary_df = combined_diagnoses_procedures_df.join(alphanumeric_tokens.select("HADM_ID").distinct(),
                                                                                     on='HADM_ID', how='inner')
    ICD_index_dict = {}
    unique_ICD9_Codes = combined_diagnoses_procedures_summary_df.select(
        "cleaned_ICD").distinct().toPandas()
    for i, val in enumerate(unique_ICD9_Codes.cleaned_ICD.unique()):
        ICD_index_dict[val] = i
    reverse_ICD_index_dict = {val: ind for ind, val in ICD_index_dict.items()}
    # https://mimic.physionet.org/mimictables/admissions/
    # Since each unique hospital visit for a patient is assigned a unique HADM_ID,the ADMISSIONS table can be considered as a definition table for HADM_ID
    pivot_combined_diagnoses_procedures_summary_df = combined_diagnoses_procedures_summary_df.select(["SUBJECT_ID", "HADM_ID", "cleaned_ICD"])\
        .groupBy(["SUBJECT_ID", "HADM_ID"])\
        .agg(F.collect_set("cleaned_ICD"))\
        .withColumnRenamed("collect_set(cleaned_ICD)", "Target")

    modelling_data = alphanumeric_tokens.select(["SUBJECT_ID", "HADM_ID", "alphanumeric"])\
        .join(pivot_combined_diagnoses_procedures_summary_df, on=["SUBJECT_ID", "HADM_ID"], how="inner")\
        .select(['alphanumeric', 'Target'])

    # (Pdb) modelling_data.show(1)
    # +----------+-------+--------------------+--------------------+
    # |SUBJECT_ID|HADM_ID|        alphanumeric|              Target|
    # +----------+-------+--------------------+--------------------+
    # |     10661| 139315|[admission, date,...|[272.0, 01.39, 78...|
    # +----------+-------+--------------------+--------------------+
    # only showing top 1 row

    # word2Vec_initialized = Word2Vec(vectorSize=embedding_size, minCount=3,
    #                                 maxSentenceLength=2500, inputCol="alphanumeric", outputCol="word2vec_features")
    word2Vec_initialized = Word2Vec(vectorSize=embedding_size, minCount=3,
                                    inputCol="alphanumeric", outputCol="word2vec_features")
    word2Vec_saved = word2Vec_initialized.fit(modelling_data)
    # (Pdb) word2Vec_saved.getVectors().printSchema()
    # root
    # |-- word: string (nullable = true)
    # |-- vector: vector (nullable = true)
    # model.getVectors()
    # (Pdb) word2Vec_saved.getVectors().show(truncate=False)
    # +-------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    # |word         |vector                                                                                                                                              |
    # +-------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    # |16mmhg       |[0.7607946991920471,0.5120894312858582,-0.039861034601926804,0.9393578171730042,0.48038697242736816,0.42862483859062195,-0.16968980431556702]       |
    # |professed    |[-0.09192302823066711,0.08201824873685837,-0.042592667043209076,0.07711202651262283,-0.10132941603660583,-0.05493719503283501,-0.010550925508141518]|
    # |laryngoscope |[-0.0775415375828743,-0.11543910950422287,-0.3991227149963379,0.7404215335845947,-0.22687077522277832,0.37258172035217285,0.24691277742385864]      |
    # |spcx         |[-0.05877019464969635,0.3695179224014282,0.5838702917098999,0.8910728096961975,-0.3550351560115814,0.1966133713722229,0.8650023937225342]           |
    # |pathogens    |[0.5665009617805481,0.07116593420505524,-0.10976329445838928,0.7977755069732666,-0.3702443838119507,-0.2615824043750763,0.3329234719276428]         |
    # |6x8cm        |[0.026608457788825035,0.0865047499537468,-0.11708809435367584,0.05032869428396225,0.016296658664941788,-0.009937587194144726,0.23116841912269592]   |
    # |metacarpals  |[0.1258579045534134,-0.010739175602793694,-0.2998082637786865,0.3463990390300751,0.17682984471321106,0.16120681166648865,0.3072815537452698]        |
    # |ihss         |[-0.0041815973818302155,0.37178319692611694,-0.06204785779118538,0.5714012384414673,0.6353005766868591,-0.058466531336307526,-0.022570976987481117] |
    # |significnt   |[0.03681206703186035,0.06724397838115692,-0.17557406425476074,0.09283604472875595,-5.426286952570081E-5,0.07902484387159348,0.024391375482082367]   |
    # |phosphates   |[-0.17698653042316437,-0.5351877212524414,1.4844105243682861,-0.03846435248851776,-0.5014886260032654,-0.30650365352630615,0.9576427340507507]      |
    # |rpls         |[0.07899224013090134,0.1276223063468933,-0.11721231788396835,0.16966769099235535,0.21127955615520477,-0.020494528114795685,0.055421482771635056]    |
    # |antithymocyte|[-0.0231438260525465,0.03851597383618355,0.025305867195129395,0.19898726046085358,0.007607527542859316,-0.18684609234333038,0.0734134316444397]     |
    # |incident     |[0.2986122965812683,0.07568827271461487,-0.5915249586105347,0.520882248878479,-0.3056448996067047,-0.5316665172576904,-0.34297266602516174]         |
    # |serious      |[0.013321951031684875,-0.43762344121932983,-0.8650598526000977,0.4826090335845947,-0.11169228702783585,-0.6041423678398132,0.35679540038108826]     |
    # |wgbh         |[-0.13728898763656616,-0.002356042619794607,-0.045427873730659485,0.041550442576408386,0.1041225716471672,-0.1486842930316925,0.020557235926389694] |
    # |bronschoscopy|[-0.03940870985388756,-0.04155345261096954,-0.17778745293617249,0.024293195456266403,0.18853092193603516,0.08649323135614395,-0.014987383037805557] |
    # |especailly   |[0.10230565071105957,-0.04630622640252113,-0.08533389866352081,-0.019424952566623688,-0.06311855465173721,0.036639533936977386,0.10097602754831314] |
    # |tunelled     |[0.0906904861330986,0.022048281505703926,-0.10217021405696869,0.34616929292678833,0.28525111079216003,0.04009472578763962,0.05900655686855316]      |
    # |satts        |[0.10590466111898422,0.31600508093833923,-0.18212538957595825,0.06449481844902039,-0.0896698534488678,-0.11846103519201279,-0.10716819763183594]    |
    # |comply       |[-0.21659451723098755,-0.007872184738516808,-0.6273578405380249,0.3926701843738556,-0.856886088848114,-0.2883334159851074,0.08432874828577042]      |
    # +-------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    # only showing top 20 rows
    # word2Vec_saved.overwrite().save(word2vec_embedding_location)

    # https://stackoverflow.com/questions/38384347/how-to-split-vector-into-columns-using-pyspark
    # word2Vec_saved.transform(train_df).withColumn("concat_text",concat_ws(" ",col("alphanumeric")))\
    #     .withColumn("concat_text",col("alphanumeric")
    # https://github.com/Azure-Samples/MachineLearningSamples-BiomedicalEntityExtraction/blob/master/code/02_modeling/01_feature_engineering/2_Train_Word2Vec_Model_Spark.py#L191

    word_vectors = word2Vec_saved.getVectors()
    
    # word_vectors.createOrReplaceTempView("word2vec_spark")
    word_vectors_copy = word_vectors
    get_ith_column = udf(get_ith_column_, DoubleType())
    for i in range(embedding_size):
        word_vectors_copy = word_vectors_copy.withColumn(
            "col"+str(i), get_ith_column("vector", lit(i)))

    word_vectors_copy = word_vectors_copy.drop("vector")
    # +------+--------------------+-------------------+------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
    # |  word|              vector|               col0|              col1|               col2|               col3|               col4|               col5|               col6|
    # +------+--------------------+-------------------+------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
    # |16mmhg|[-0.3276940882205...|-0.3276940882205963|0.3549065589904785|0.49013954401016235|0.47031646966934204|-0.0979413092136383|-0.9541968107223511|0.02784506045281887|
    # +------+--------------------+-------------------+------------------+-------------------+-------------------+-------------------+-------------------+-------------------+

    # need to convert to index, else cant convert to tensor.
    embedding_layer = torch.zeros(
        (word_vectors_copy.count()+2, len(word_vectors_copy.columns)-1))
    temp = word_vectors_copy.toPandas()
    vocab_dict = {}
    # for padding
    embedding_layer[0, :] = torch.zeros((1, len(word_vectors_copy.columns)-1))
    # for unknown
    embedding_layer[-1, :] = torch.rand((1, len(word_vectors_copy.columns)-1))
    for i, val in enumerate(temp.to_numpy()):
        vocab_dict[val[0]] = i+1
        embedding_layer[i+1, :] = torch.tensor(val[1:].astype("float32"))

    reverse_vocab_dict = {val: ind for ind, val in vocab_dict.items()}
    # vocab.write.mode("overwrite").parquet(f"{server}/intermediate/vocab")
    # Test Train split
    # https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html

    # filter_length_udf = udf(lambda row: [x for x in row if not is_numeric(x)], ArrayType(StringType()))
    # alphanumeric_tokens = tokenized.withColumn('alphanumeric', filter_length_udf(col('words')))

    get_index_udf = udf(get_index, ArrayType(IntegerType()))
    modelling_data_indexed = modelling_data.withColumn(
        "alphanumeric_index_of_words", get_index(vocab_dict)(F.col("alphanumeric")))
    # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    modelling_data_indexed = modelling_data_indexed.withColumn("Target_index_of_words", get_index(ICD_index_dict)(F.col("Target")))\
        .select(["alphanumeric_index_of_words", "Target_index_of_words"])
    train_df, test_df, valid_df = modelling_data_indexed.randomSplit(
        [0.80, 0.10, 0.10], seed=41)

    # MyVariableRNN(num_features)
    train_df.write.mode("overwrite").parquet(f"{server}/intermediate/train_df")
    test_df.write.mode("overwrite").parquet(f"{server}/intermediate/test_df")
    valid_df.write.mode("overwrite").parquet(f"{server}/intermediate/valid_df")

    with open(vocab_dict_file, 'wb') as handle:
        pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reverse_vocab_dict_file, 'wb') as handle:
        pickle.dump(reverse_vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(embedding_layer_file, 'wb') as handle:
        pickle.dump(embedding_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(icd9_index_dict_file, 'wb') as handle:
        pickle.dump(ICD_index_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(reverse_icd9_index_dict_file, 'wb') as handle:
        pickle.dump(reverse_icd9_index_dict_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # (Pdb) train_df.count()
    # 39672
    # (Pdb) test_df.count()
    # 10350
    # (Pdb) valid_df.count()
    # 2700

    # https://stackoverflow.com/questions/58995226/one-hot-encode-of-multiple-string-categorical-features-using-spark-dataframes

    # word2vec on train dataset

    # https://spark.apache.org/docs/latest/ml-features#word2vec
    # https://discuss.pytorch.org/t/how-to-create-a-1d-sparse-tensors-from-given-list-of-indices-and-values/90415
    # https://stackoverflow.com/questions/53272749/why-does-sparks-word2vec-return-a-vector/53529436

    # Model Training
    # https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321

def generate_dataset_splits(split_name):
    train_df = spark.read.parquet(f"{server}/intermediate/{split_name}")
    with open(embedding_layer_file, 'rb') as f:
        embedding_layer = pickle.load(f)

    train_df_driver = train_df.toPandas()
    
    train_seqs = []
    train_seqs_length = []
    for sentences in train_df_driver["alphanumeric_index_of_words"].values:
        list_of_words = []
        count = 0
        sentenced_formatted = sentences[1:-1].split(",")
        for words in sentenced_formatted:
            # removing low frequency words rejected by word2vec
            non_nan = int(words.strip())
            if non_nan != -1:
                # Fix for batch size 1
                list_of_words.append([non_nan])
            else:
                # unknown token - index of embedding layer starts from 0
                list_of_words.append([len(embedding_layer)-1])
            count = count+1
            if count >= 1000:
                break
        train_seqs.append(list_of_words)
        train_seqs_length.append(count)
    
    # print(len(train_seqs))
    # seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
    
    train_labels = []
    for target_row in train_df_driver["Target_index_of_words"].values:
        targets = target_row[1:-1].split(",")
        train_labels.append([int(x) for x in targets])
    
    seqs_file = split_name + '_seqs.pkl'
    labels_file = split_name + '_labels.pkl'
    with open(seqs_file, 'wb') as handle:
        pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(labels_file, 'wb') as handle:
        pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return seqs_file,labels_file
        # Missing ICD9 code check
        # -1 in [y for x in train_labels for y in x]
        # (Pdb) train_labels[0]
        # [2961, 6209, 2462, 2875, 2468, 642, 5094, 6074, 2187]

        # valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
        # valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
        # test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
        # test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

        # TODO: Need to change this function - as we have used all vocab for word2vec, array size should not be on train
        # num_features = calculate_num_features(train_seqs)
def train_model(seqs_file,labels_file,model_name):
    # num_features = len(vocab_dict)
    
    with open(seqs_file[0], 'rb') as f:
        train_seqs = pickle.load(f)
    with open(labels_file[0], 'rb') as f:
        train_labels = pickle.load(f)

    with open(seqs_file[1], 'rb') as f:
        valid_seqs = pickle.load(f)
    with open(labels_file[1], 'rb') as f:
        valid_labels = pickle.load(f)

    with open(seqs_file[2], 'rb') as f:
        test_seqs = pickle.load(f)
    with open(labels_file[2], 'rb') as f:
        test_labels = pickle.load(f)

    with open(embedding_layer_file, 'rb') as f:
        embedding_layer = pickle.load(f)
    with open(vocab_dict_file, 'rb') as f:
        vocab_dict = pickle.load(f)
    with open(icd9_index_dict_file, 'rb') as f:
        ICD_index_dict = pickle.load(f)
    num_features = len(vocab_dict)
    train_dataset = VisitSequenceWithLabelDataset(
        train_seqs, train_labels, num_features, num_categories=len(ICD_index_dict))
    valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features, num_categories=len(ICD_index_dict))
    test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features, num_categories=len(ICD_index_dict))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
    if model_name=='GRU':
        model = mymodels.testGRU(
            weights_matrix=embedding_layer, num_categories=len(ICD_index_dict))
        save_file = 'GRU.pth'
    elif model_name=='CNN_Attn':
        model=mymodels.CNNAttn(num_output_categories=len(ICD_index_dict),weights_matrix=embedding_layer,num_filters=50,kernel_size=3)
        save_file = 'CNN_Attn.pth'
    elif model_name=="LSTM":
        model = mymodels.testLSTM(
            weights_matrix=embedding_layer, num_categories=len(ICD_index_dict))
        save_file = 'LSTM.pth'
    else:
        sys.exit("Model Not Available")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)
    # https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch}")
        train_loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        if is_best:
            best_val_acc = valid_accuracy
            # https://piazza.com/class/ki87klxs9yite?cid=397_f2
            # torch.save(model, os.path.join(PATH_OUTPUT, save_file))
            torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies,model_name)

    best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

    # individual_visit_sequence.extend(x for x in a if x is not None)

def is_numeric(x):
    if x:
        return x.isdigit()
    else:
        # print("error in is_numeric")
        return False


def get_ith_column_(val, index):
    try:
        return float(val[index])
    except ValueError:
        return None

# https://mungingdata.com/pyspark/udf-dict-broadcast/

# target should be long..but indices for embedding should be float
def get_index(vocab_dict):
    def f(x):
        return [vocab_dict.get(i, -1) for i in x]
    return F.udf(f)


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features, num_categories):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        # https://stackoverflow.com/questions/50981714/multi-label-multi-class-image-classifier-convnet-with-pytorch
        # https://stackoverflow.com/questions/56123419/how-to-cover-a-label-list-under-the-multi-label-classification-context-into-one

        # how to deal with zero index
        one_hot_labels = torch.zeros(
            size=(len(labels), num_categories), dtype=torch.float32)
        for i, label in enumerate(labels):
            label = torch.LongTensor(label)
            one_hot_labels[i] = one_hot_labels[i].scatter_(
                dim=0, index=label, value=1.)
        # (Pdb) one_hot_labels[0]
        # tensor([0., 0., 0.,  ..., 0., 0., 0.])
        # (Pdb) one_hot_labels[0].sum()
        # tensor(9.)
        # (Pdb) len(labels[0])
        # 9
        # (Pdb) one_hot_labels[0].size()
        # torch.Size([8929])
        # (Pdb

        self.labels = one_hot_labels

        # TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
        # import pdb;pdb.set_trace()
        # Wewill store the labels as a List of integer labels
        # It is already a list

        # TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.

        # the sequence data as a List of matrix whose i-th row represents
        # i-th visit while j-th column corresponds to the integer feature ID j
        # (Pdb) seqs[0]
        # [[78, 8, 34, 26], [73, 74, 75, 8, 57, 76, 77, 26]]
        # numpy array column width with num_features
        # how to mark them as one? loop in loop? very inefficient
        # embedding layer needs only indices
        self.seqs = seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
    where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

    :returns
            seqs (FloatTensor) - 3D of batch_size X max_length X num_features
            lengths (LongTensor) - 1D of batch_size
            labels (LongTensor) - 1D of batch_size
    """
    
    # TODO: Return the following two things
    # TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
    # TODO: 2. Tensor contains the label of each sequence

    # Tensor that consists of matrices with the same number of rows (visits) by padding zero-rows at the end of matrices
    # shorter than the largest matrix in the mini-batch.
    # seqs_tensor = torch.FloatTensor()
    # lengths_tensor = torch.LongTensor()
    # labels_tensor = torch.LongTensor()

    # return (seqs_tensor, lengths_tensor), labels_tensor
    # 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
    visit_sequence, label = zip(*batch)
    # visit_sequence
    # (array([[0., 0., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[0., 1., 0., ... 0., 0.]]), array([[0., 1., 0., ... 0., 0.]]), array([[0., 1., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[0., 1., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), array([[1., 0., 0., ... 0., 0.]]), array([[0., 0., 0., ... 0., 0.]]), ...)
    # len(visit_sequence)
    # 32
    # len(label)
    # 32
    # batch[29][0].shape
    # (5, 911)
    visit_max = [len(x) for x in visit_sequence]

    max_len = max(visit_max)

    combined = [(len(visit_sequence), visit_sequence, label)
                for visit_sequence, label in batch]
    sorted_combined = sorted(combined, key=lambda x: float(x[0]), reverse=True)

    # creates a mini-batch represented by a 3D
    # Tensor that consists of matrices with the same number of rows (visits) by padding zero-rows
    # at the end of matrices shorter than the largest matrix in the mini-batch.
    # 3D of batch_size X max_length X num_features
    # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    np_array_column = 1  # 1D array
    seqs = torch.zeros((len(batch), max_len, np_array_column))
    lengths = torch.zeros(len(batch))
    # this needs to be changed for multi label
    # this breaks if batch size is 1
    labels = torch.zeros(len(batch), len(label[0]))
    # Also, the order
    # of matrices in the Tensor must be sorted by the length of visits in descending order.
    for i, sorted_individual_row in enumerate(sorted_combined):
        (count_of_visit_sequence, individual_visit_sequence,
        individual_label) = sorted_individual_row
        filled_rows = len(individual_visit_sequence)
        fixed_array = np.append(\
            np.array([np.array(x) for x in individual_visit_sequence]), \
            np.zeros((max_len - filled_rows, np_array_column)),\
                axis=0)
    
        seqs[i] =  torch.tensor(fixed_array)
        lengths[i] = torch.tensor(count_of_visit_sequence)
        labels[i] = individual_label
    # indexes should be longtype int64
    # BCEwithLogitsLoss want float, not long
    return (seqs.long(), lengths.long()), labels.float()

if __name__ == '__main__':
    run_data_processing=False
    if run_data_processing:
        data_processing()
    train_seqs_file,train_labels_file=generate_dataset_splits('train_df')
    valid_seqs_file,valid_labels_file=generate_dataset_splits('valid_df')
    test_seqs_file,test_labels_file=generate_dataset_splits('test_df')
    seqs_file=(train_seqs_file,valid_seqs_file,test_seqs_file)
    labels_file=(train_labels_file,valid_labels_file,test_labels_file)
    train_model(seqs_file,labels_file,model_name='CNN_Attn')
    train_model(seqs_file,labels_file,model_name='GRU')
    train_model(seqs_file,labels_file,model_name='LSTM')

    # try:
    #     for i, sorted_individual_row in enumerate(sorted_combined):
    #         (count_of_visit_sequence, individual_visit_sequence,
    #         individual_label) = sorted_individual_row
    #         filled_rows = len(individual_visit_sequence)
    #         a = [0]*(max_len-filled_rows)
    #         individual_visit_sequence.extend(x for x in a if x is not None)
    #         # convert this to torch - forcing int64
    #         seqs[i] =  torch.tensor(individual_visit_sequence).to(torch.int64)
    #         lengths[i] = torch.tensor(count_of_visit_sequence)
    #         labels[i] = individual_label
    # except:
    #     import pdb;pdb.set_trace()
    # # indexes should be longtype int64
    # return (seqs, lengths.long()), labels.long()