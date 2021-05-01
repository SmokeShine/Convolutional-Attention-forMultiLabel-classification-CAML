#!/usr/bin/env python
import argparse
import logging
import logging.config
import os
import sys
import pickle
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, lit
from pyspark.sql import functions as F
from pyspark.ml.feature import *
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sql_query import diagnosis_query, procedures_query
# Util function from assignment
from utils import train, evaluate
import mymodels
from plots import plot_learning_curves
spark = SparkSession.builder.appName('BigDataHealthSparkApp').getOrCreate()
VOCAB_DICT_FILE = "vocab_dict.pkl"
EMBEDDING_LAYER_FILE = "embedding_layer.pkl"
REVERSE_VOCAB_DICT_FILE = "reverse_vocab_dict.pkl"
ICD9_INDEX_DICT_FILE = "icd9_dict_index.pkl"
REVERSE_ICD9_INDEX_DICT_FILE = "reverse_icd9_dict_index.pkl"
PATH_OUTPUT = "./"
logging.config.fileConfig("logging.ini")
logger = logging.getLogger()
torch.manual_seed(1)


def data_processing():
    """[summary]
    """
    note_events_df, diagnoses_icd_df, procedures_icd_df = load_data()
    # https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/HospitalQualityInits/Downloads/HospitalAppendix_F.pdf
    diagnoses_icd_df.registerTempTable("diagnoses_icd_df")
    diagnoses_icd_df = spark.sql(diagnosis_query)
    procedures_icd_df.registerTempTable("procedures_icd_df")
    procedures_icd_df = spark.sql(procedures_query)
    combined_diagnoses_procedures_df = diagnoses_icd_df.union(
        procedures_icd_df)
    note_events_df = note_events_df.orderBy(
        ["HADM_ID", "CHARTTIME"]).dropDuplicates(subset=['HADM_ID'])
    discharge_summary_df = note_events_df.filter(
        note_events_df["CATEGORY"] == "Discharge summary")
    # https://spark.apache.org/docs/latest/ml-features#tokenizer
    # https://stackoverflow.com/questions/6053541/regex-every-non-alphanumeric-character-except-white-space-or-colon
    # https://stackoverflow.com/questions/48278489/how-to-create-a-custom-tokenizer-in-pyspark-ml/48279714
    regexTokenizer = RegexTokenizer(
        inputCol="TEXT", outputCol="words", pattern=r'\w+', gaps=False)
    tokenized = regexTokenizer.transform(discharge_summary_df)
    # https://stackoverflow.com/questions/53951215/how-do-i-remove-words-numerics-pyspark
    is_numeric_udf = udf(is_numeric, BooleanType())
    filter_length_udf = udf(
        lambda row: [x for x in row if not is_numeric(x)], ArrayType(StringType()))
    alphanumeric_tokens = tokenized.withColumn(
        'alphanumeric', filter_length_udf(col('words')))
    combined_diagnoses_procedures_summary_df = combined_diagnoses_procedures_df.join(
        alphanumeric_tokens.select("HADM_ID").distinct(),
        on='HADM_ID', how='inner')
    icd_index_dict = {}
    unique_icd9_codes = combined_diagnoses_procedures_summary_df.select(
        "cleaned_ICD").distinct().toPandas()
    for i, val in enumerate(unique_icd9_codes.cleaned_ICD.unique()):
        icd_index_dict[val] = i
    reverse_icd_index_dict = {val: ind for ind, val in icd_index_dict.items()}
    # https://mimic.physionet.org/mimictables/admissions/
    # Since each unique hospital visit for a patient is assigned a unique HADM_ID,
    # the ADMISSIONS table can be considered as a definition table for HADM_ID
    pivot_combined_diagnoses_procedures_summary_df = combined_diagnoses_procedures_summary_df\
        .select(["SUBJECT_ID", "HADM_ID", "cleaned_ICD"])\
        .groupBy(["SUBJECT_ID", "HADM_ID"])\
        .agg(F.collect_set("cleaned_ICD"))\
        .withColumnRenamed("collect_set(cleaned_ICD)", "Target")
    modelling_data = alphanumeric_tokens.select(["SUBJECT_ID", "HADM_ID", "alphanumeric"])\
        .join(pivot_combined_diagnoses_procedures_summary_df,
              on=["SUBJECT_ID", "HADM_ID"], how="inner")\
        .select(['alphanumeric', 'Target'])
    word2Vec_initialized = Word2Vec(vectorSize=EMBEDDING_SIZE, minCount=3,
                                    inputCol="alphanumeric", maxSentenceLength=2500,
                                    outputCol="word2vec_features")
    word2Vec_saved = word2Vec_initialized.fit(modelling_data)
    # https://stackoverflow.com/questions/38384347/how-to-split-vector-into-columns-using-pyspark
    # https://github.com/Azure-Samples/MachineLearningSamples-BiomedicalEntityExtraction/blob/master/code/02_modeling/01_feature_engineering/2_Train_Word2Vec_Model_Spark.py#L191
    word_vectors = word2Vec_saved.getVectors()
    word_vectors_copy = word_vectors
    get_ith_column = udf(get_ith_column_, DoubleType())
    for i in range(EMBEDDING_SIZE):
        word_vectors_copy = word_vectors_copy.withColumn(
            "col"+str(i), get_ith_column("vector", lit(i)))
    word_vectors_copy = word_vectors_copy.drop("vector")
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
    # https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html
    get_index_udf = udf(get_index, ArrayType(IntegerType()))
    modelling_data_indexed = modelling_data.withColumn(
        "alphanumeric_index_of_words", get_index(vocab_dict)(F.col("alphanumeric")))
    # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    modelling_data_indexed = modelling_data_indexed\
        .withColumn("Target_index_of_words",
                    get_index(icd_index_dict)(F.col("Target")))\
        .select(["alphanumeric_index_of_words", "Target_index_of_words"])
    train_df, test_df, valid_df = modelling_data_indexed.randomSplit(
        [0.80, 0.10, 0.10], seed=41)
    train_df.write.mode("overwrite").parquet(f"{SERVER}/intermediate/train_df")
    test_df.write.mode("overwrite").parquet(f"{SERVER}/intermediate/test_df")
    valid_df.write.mode("overwrite").parquet(f"{SERVER}/intermediate/valid_df")
    with open(VOCAB_DICT_FILE, 'wb') as handle:
        pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(REVERSE_VOCAB_DICT_FILE, 'wb') as handle:
        pickle.dump(reverse_vocab_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(EMBEDDING_LAYER_FILE, 'wb') as handle:
        pickle.dump(embedding_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ICD9_INDEX_DICT_FILE, 'wb') as handle:
        pickle.dump(icd_index_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(REVERSE_ICD9_INDEX_DICT_FILE, 'wb') as handle:
        pickle.dump(reverse_icd_index_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_data():
    """[summary]
    Returns:
        [type]: [description]
    """
    note_events_df = spark.read.format("csv").\
        option("header", True)\
        .option("multiline", "true")\
        .option("quote", '"')\
        .option("escape", '"')\
        .load(NOTE_EVENT_DATA_LOCATION)
    diagnoses_icd_df = spark.read.format("csv").\
        option("header", True).\
        load(DIAGNOSES_ICD_DATA_LOCATION)
    procedures_icd_df = spark.read.format("csv").\
        option("header", True).\
        load(PROCEDURES_ICD_DATA_LOCATION)
    return note_events_df, diagnoses_icd_df, procedures_icd_df
    # https://stackoverflow.com/questions/58995226/one-hot-encode-of-multiple-string-categorical-features-using-spark-dataframes
    # https://spark.apache.org/docs/latest/ml-features#word2vec
    # https://discuss.pytorch.org/t/how-to-create-a-1d-sparse-tensors-from-given-list-of-indices-and-values/90415
    # https://stackoverflow.com/questions/53272749/why-does-sparks-word2vec-return-a-vector/53529436
    # Model Training
    # https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321


def generate_dataset_splits(split_name):
    """[summary]
    Args:
        split_name ([type]): [description]
    Returns:
        [type]: [description]
    """
    train_df = spark.read.parquet(f"{SERVER}/intermediate/{split_name}")
    with open(EMBEDDING_LAYER_FILE, 'rb') as file_name:
        embedding_layer = pickle.load(file_name)
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
            if count >= MAX_LENGTH_OF_SENTENCE:
                break
        train_seqs.append(list_of_words)
        train_seqs_length.append(count)
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
    return seqs_file, labels_file


def train_model(seqs_file, labels_file, model_name, topk):
    """[summary]
    Args:
        seqs_file ([type]): [description]
        labels_file ([type]): [description]
        model_name ([type]): [description]
    Returns:
        [type]: [description]
    """
    train_seqs, train_labels,\
        valid_seqs, valid_labels, \
        test_seqs, test_labels, \
        embedding_layer, \
        vocab_dict, icd_index_dict = load_pickle(seqs_file, labels_file)
    num_features = len(vocab_dict)
    train_dataset = VisitSequenceWithLabelDataset(
        train_seqs, train_labels, num_features, num_categories=len(icd_index_dict))
    valid_dataset = VisitSequenceWithLabelDataset(
        valid_seqs, valid_labels, num_features, num_categories=len(icd_index_dict))
    test_dataset = VisitSequenceWithLabelDataset(
        test_seqs, test_labels, num_features, num_categories=len(icd_index_dict))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
    if model_name == 'GRU':
        model = mymodels.testGRU(
            weights_matrix=embedding_layer, num_categories=len(icd_index_dict))
        save_file = 'GRU.pth'
    elif model_name == 'CNN_Attn':
        model = mymodels.CNNAttn(num_output_categories=len(
            icd_index_dict), weights_matrix=embedding_layer, num_filters=50, kernel_size=3)
        save_file = 'CNN_Attn.pth'
    elif model_name == "LSTM":
        model = mymodels.testLSTM(
            weights_matrix=embedding_layer, num_categories=len(icd_index_dict))
        save_file = 'LSTM.pth'
    else:
        sys.exit("Model Not Available")
    logger.info(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    # https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    train_micro_f1s, train_micro_f1s = [], []
    valid_micro_f1s, valid_micro_f1s = [], []
    train_macro_f1s, train_macro_f1s = [], []
    valid_macro_f1s, valid_macro_f1s = [], []
    train_hammings, train_hammings = [], []
    valid_hammings, valid_hammings = [], []
    train_micro_auc_rocs, train_micro_auc_rocs = [], []
    valid_micro_auc_rocs, valid_micro_auc_rocs = [], []
    train_macro_auc_rocs, train_macro_auc_rocs = [], []
    valid_macro_auc_rocs, valid_macro_auc_rocs = [], []
    early_stopping_counter = 0
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch}")
        train_loss, train_package = train(
            logger, model, device, train_loader, criterion, optimizer, epoch)
        train_accuracy = train_package[0]
        train_micro_f1 = train_package[1]
        train_macro_f1 = train_package[2]
        train_hamming = train_package[3]
        train_micro_auc_roc = train_package[4]
        train_macro_auc_roc = train_package[5]
        valid_loss, valid_package, valid_results = evaluate(
            logger, model, device, valid_loader, criterion, topk)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracy = valid_package[0]
        valid_micro_f1 = valid_package[1]
        valid_macro_f1 = valid_package[2]
        valid_hamming = valid_package[3]
        valid_micro_auc_roc = valid_package[4]
        valid_macro_auc_roc = valid_package[5]
        train_accuracies.append(train_accuracy)
        train_micro_f1s.append(train_micro_f1)
        train_macro_f1s.append(train_macro_f1)
        train_hammings.append(train_hamming)
        train_micro_auc_rocs.append(train_micro_auc_roc)
        train_macro_auc_rocs.append(train_macro_auc_roc)
        valid_accuracies.append(valid_accuracy)
        valid_micro_f1s.append(valid_micro_f1)
        valid_macro_f1s.append(valid_macro_f1)
        valid_hammings.append(valid_hamming)
        valid_micro_auc_rocs.append(valid_micro_auc_roc)
        valid_macro_auc_rocs.append(valid_macro_auc_roc)
        is_best = valid_macro_auc_roc > best_val_acc
        if is_best:
            early_stopping_counter = 0
            best_val_acc = valid_macro_auc_roc
            # https://piazza.com/class/ki87klxs9yite?cid=397_f2
            torch.save(model, os.path.join(PATH_OUTPUT, save_file),
                       _use_new_zipfile_serialization=False)
        else:
            early_stopping_counter = early_stopping_counter+1
        if early_stopping_counter >= PATIENCE:
            # exiting training process
            break
    best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    test_loss, test_package, test_results = evaluate(
        logger, best_model, device, test_loader, criterion, topk)
    return train_losses, valid_losses, \
        train_macro_auc_rocs, \
        valid_macro_auc_rocs, test_results


def load_pickle(seqs_file, labels_file):
    """[summary]
    Args:
        seqs_file ([type]): [description]
        labels_file ([type]): [description]
    Returns:
        [type]: [description]
    """
    with open(seqs_file[0], 'rb') as file_name:
        train_seqs = pickle.load(file_name)
    with open(labels_file[0], 'rb') as file_name:
        train_labels = pickle.load(file_name)
    with open(seqs_file[1], 'rb') as file_name:
        valid_seqs = pickle.load(file_name)
    with open(labels_file[1], 'rb') as file_name:
        valid_labels = pickle.load(file_name)
    with open(seqs_file[2], 'rb') as file_name:
        test_seqs = pickle.load(file_name)
    with open(labels_file[2], 'rb') as file_name:
        test_labels = pickle.load(file_name)
    with open(EMBEDDING_LAYER_FILE, 'rb') as file_name:
        embedding_layer = pickle.load(file_name)
    with open(VOCAB_DICT_FILE, 'rb') as file_name:
        vocab_dict = pickle.load(file_name)
    with open(ICD9_INDEX_DICT_FILE, 'rb') as file_name:
        icd_index_dict = pickle.load(file_name)
    return train_seqs, train_labels,\
        valid_seqs, valid_labels,\
        test_seqs, test_labels,\
        embedding_layer,\
        vocab_dict,\
        icd_index_dict


def is_numeric(x):
    """[summary]
    Args:
        x ([type]): [description]
    Returns:
        [type]: [description]
    """
    if x:
        return x.isdigit()
    else:
        return False


def get_ith_column_(val, index):
    """[summary]
    Args:
        val ([type]): [description]
        index ([type]): [description]
    Returns:
        [type]: [description]
    """
    try:
        return float(val[index])
    except ValueError:
        return None
# https://mungingdata.com/pyspark/udf-dict-broadcast/


def get_index(vocab_dict):
    """[summary]
    Args:
        vocab_dict ([type]): [description]
    """
    def f(x):
        return [vocab_dict.get(i, -1) for i in x]
    return F.udf(f)


class VisitSequenceWithLabelDataset(Dataset):
    """[summary]
    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, seqs, labels, num_features, num_categories):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of
            codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """
        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")
        # https://stackoverflow.com/questions/50981714/multi-label-multi-class-image-classifier-convnet-with-pytorch
        # https://stackoverflow.com/questions/56123419/how-to-cover-a-label-list-under-the-multi-label-classification-context-into-one
        one_hot_labels = torch.zeros(
            size=(len(labels), num_categories), dtype=torch.float32)
        for i, label in enumerate(labels):
            label = torch.LongTensor(label)
            one_hot_labels[i] = one_hot_labels[i].scatter_(
                dim=0, index=label, value=1.)
        self.labels = one_hot_labels
        self.seqs = seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
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
    visit_sequence, label = zip(*batch)
    visit_max = [len(x) for x in visit_sequence]
    max_len = max(visit_max)
    combined = [(len(visit_sequence), visit_sequence, label)
                for visit_sequence, label in batch]
    sorted_combined = sorted(combined, key=lambda x: float(x[0]), reverse=True)
    # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    np_array_column = 1  # 1D array
    seqs = torch.zeros((len(batch), max_len, np_array_column))
    lengths = torch.zeros(len(batch))
    labels = torch.zeros(len(batch), len(label[0]))
    for i, sorted_individual_row in enumerate(sorted_combined):
        (count_of_visit_sequence, individual_visit_sequence,
         individual_label) = sorted_individual_row
        filled_rows = len(individual_visit_sequence)
        fixed_array = np.append(
            np.array([np.array(x) for x in individual_visit_sequence]),
            np.zeros((max_len - filled_rows, np_array_column)),
            axis=0)
        seqs[i] = torch.tensor(fixed_array)
        lengths[i] = torch.tensor(count_of_visit_sequence)
        labels[i] = individual_label
    return (seqs.long(), lengths.long()), labels.float()


def new_run(train_model, seqs_file, labels_file, topk):
    """[summary]
    Args:
        train_model ([type]): [description]
        seqs_file ([type]): [description]
        labels_file ([type]): [description]
    """
    logger.info("Training CNN_Attn")
    train_losses_cnn, valid_losses_cnn, \
        train_macro_auc_rocs_cnn, \
        valid_macro_auc_rocs_cnn, test_results_cnn = train_model(seqs_file, labels_file,
                                                                 model_name='CNN_Attn', topk=topk)
    logger.info("Training GRU")
    train_losses_gru, valid_losses_gru, \
        train_macro_auc_rocs_gru, \
        valid_macro_auc_rocs_gru, test_results_gru = train_model(seqs_file, labels_file,
                                                                 model_name='GRU', topk=topk)
    logger.info("Training LSTM")
    train_losses_lstm, valid_losses_lstm, \
        train_macro_auc_rocs_lstm, \
        valid_macro_auc_rocs_lstm, test_results_lstm = train_model(seqs_file, labels_file,
                                                                   model_name='LSTM', topk=topk)
    train_cnn_save = (train_losses_cnn, valid_losses_cnn,
                      train_macro_auc_rocs_cnn,
                      valid_macro_auc_rocs_cnn, test_results_cnn)
    train_gru_save = (train_losses_gru, valid_losses_gru,
                      train_macro_auc_rocs_gru,
                      valid_macro_auc_rocs_gru, test_results_gru)
    train_lstm_save = (train_losses_lstm, valid_losses_lstm,
                       train_macro_auc_rocs_lstm,
                       valid_macro_auc_rocs_lstm, test_results_lstm)
    save_losses(train_cnn_save, train_gru_save, train_lstm_save)


def save_losses(train_cnn_save, train_gru_save, train_lstm_save):
    """[summary]
    Args:
        train_cnn_save ([type]): [description]
        train_gru_save ([type]): [description]
        train_lstm_save ([type]): [description]
    """
    with open('train_save_CNN.pkl', 'wb') as file_name:
        pickle.dump(train_cnn_save, file_name)
    with open('train_save_GRU.pkl', 'wb') as file_name:
        pickle.dump(train_gru_save, file_name)
    with open('train_save_LSTM.pkl', 'wb') as file_name:
        pickle.dump(train_lstm_save, file_name)


def restore_losses():
    """[summary]
    Returns:
        [type]: [description]
    """
    with open('train_save_CNN.pkl', 'rb') as file_name:
        train_losses_cnn, valid_losses_cnn, \
            train_macro_auc_rocs_cnn, \
            valid_macro_auc_rocs_cnn, test_results_cnn = pickle.load(file_name)
    with open('train_save_GRU.pkl', 'rb') as file_name:
        train_losses_gru, valid_losses_gru, \
            train_macro_auc_rocs_gru, \
            valid_macro_auc_rocs_gru, test_results_gru = pickle.load(file_name)
    with open('train_save_LSTM.pkl', 'rb') as file_name:
        train_losses_lstm, valid_losses_lstm, \
            train_macro_auc_rocs_lstm, \
            valid_macro_auc_rocs_lstm, test_results_lstm = pickle.load(
                file_name)
    return train_losses_cnn, valid_losses_cnn, train_macro_auc_rocs_cnn, valid_macro_auc_rocs_cnn, test_results_cnn, \
        train_losses_gru, valid_losses_gru, train_macro_auc_rocs_gru, valid_macro_auc_rocs_gru, test_results_gru, \
        train_losses_lstm, valid_losses_lstm, train_macro_auc_rocs_lstm, valid_macro_auc_rocs_lstm, test_results_lstm


def charts(train_losses_cnn, valid_losses_cnn,
           train_macro_auc_rocs_cnn, valid_macro_auc_rocs_cnn,
           train_losses_gru, valid_losses_gru,
           train_macro_auc_rocs_gru, valid_macro_auc_rocs_gru,
           train_losses_lstm, valid_losses_lstm,
           train_macro_auc_rocs_lstm, valid_macro_auc_rocs_lstm):
    """[summary]
    Args:
        train_losses_cnn ([type]): [description]
        valid_losses_cnn ([type]): [description]
        train_macro_auc_rocs_cnn ([type]): [description]
        valid_macro_auc_rocs_cnn ([type]): [description]
        train_losses_gru ([type]): [description]
        valid_losses_gru ([type]): [description]
        train_macro_auc_rocs_gru ([type]): [description]
        valid_macro_auc_rocs_gru ([type]): [description]
        train_losses_lstm ([type]): [description]
        valid_losses_lstm ([type]): [description]
        train_macro_auc_rocs_lstm ([type]): [description]
        valid_macro_auc_rocs_lstm ([type]): [description]
    """
    plot_learning_curves(train_losses_cnn, valid_losses_cnn,
                         train_macro_auc_rocs_cnn,
                         valid_macro_auc_rocs_cnn, "CNN_Attn")
    plot_learning_curves(train_losses_gru, valid_losses_gru,
                         train_macro_auc_rocs_gru,
                         valid_macro_auc_rocs_gru, "GRU")
    plot_learning_curves(train_losses_lstm, valid_losses_lstm,
                         train_macro_auc_rocs_lstm,
                         valid_macro_auc_rocs_lstm, "LSTM")


def sequencer(generate_dataset_splits):
    """[summary]
    Args:
        generate_dataset_splits ([type]): [description]
    Returns:
        [type]: [description]
    """
    train_seqs_file, train_labels_file = generate_dataset_splits('train_df')
    valid_seqs_file, valid_labels_file = generate_dataset_splits('valid_df')
    test_seqs_file, test_labels_file = generate_dataset_splits('test_df')
    seqs_file = (train_seqs_file, valid_seqs_file, test_seqs_file)
    labels_file = (train_labels_file, valid_labels_file, test_labels_file)
    return seqs_file, labels_file


def parse_args():
    parser = argparse.ArgumentParser(description="Final Group Project",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server", nargs='?', type=str,
                        default='hdfs://localhost:9000', help="Provide address of server")
    parser.add_argument("--data_processing", action='store_true',
                        default=False, help="Data Preprocessing")
    parser.add_argument("--gpu", action='store_true',
                        default=False, help="Use GPU for training")
    parser.add_argument("--train", action='store_true',
                        default=False, help="Train Model")
    parser.add_argument("--embedding_size", nargs='?',
                        type=int, default=100, help="Size of embedding layer")
    parser.add_argument("--batch_size", nargs='?', type=int,
                        default=32, help="Batch size for training the model")
    parser.add_argument("--num_workers", nargs='?', type=int,
                        default=5, help="Number of Available CPUs")
    parser.add_argument("--num_epochs", nargs='?', type=int,
                        default=10, help="Number of Epochs for training the model")
    parser.add_argument("--patience", nargs='?', type=int,
                        default=2, help="Number of epochs Early Stopping")
    parser.add_argument("--max_length_of_sentence", nargs='?', type=int,
                        default=1000, help="Maximum length of sentence for Spark Word2Vec Model")
    parser.add_argument("--top_k", nargs='?', type=int,
                        default=5, help="Top k predictions")
    parser.add_argument("--prediction_samples", nargs='?', type=int,
                        default=5, help="Number of prediction samples")
    return parser.parse_args()


def top_k_predictions(logger, prediction_samples, test_results_cnn,
                      test_results_gru,
                      test_results_lstm,
                      reverse_icd_index_dict,
                      reverse_vocab_dict,
                      sample_test):
    for sample_test_, sample_cnn, sample_gru, sample_lstm in zip(sample_test[0:prediction_samples],
                                                                 test_results_cnn[0:prediction_samples],
                                                                 test_results_gru[0:prediction_samples],
                                                                 test_results_lstm[0:prediction_samples]):
        sentence = ' '.join(
            reverse_vocab_dict.get(word,"-") for word_list in sample_test_ for word in word_list)
        cnn_prediction = ','.join(
            reverse_icd_index_dict.get(code,"-") for code in sample_cnn)
        gru_prediction = ','.join(
            reverse_icd_index_dict.get(code,"-") for code in sample_gru)
        lstm_prediction = ','.join(
            reverse_icd_index_dict.get(code,"-") for code in sample_lstm)
        logger.info(f"{sentence[0:100]}...")
        logger.info("ICD9_CODE Predictions")
        logger.info(f"cnn_prediction:\t{cnn_prediction}")
        logger.info(f"gru_prediction:\t{gru_prediction}")
        logger.info(f"lstm_prediction:\t{lstm_prediction}\n")
        


if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    global BATCH_SIZE, EMBEDDING_SIZE,\
        USE_CUDA, MAX_LENGTH_OF_SENTENCE,\
        NUM_EPOCHS, NUM_WORKERS, SERVER, PATIENCE
    global NOTE_EVENT_DATA_LOCATION, DIAGNOSES_ICD_DATA_LOCATION
    global PROCEDURES_ICD_DATA_LOCATION, WORD2VEC_EMBEDDING_LOCATION
    SERVER = args.server
    BATCH_SIZE = args.batch_size
    NOTE_EVENT_DATA_LOCATION = f"{SERVER}/input/raw_data/NOTEEVENTS.csv"
    DIAGNOSES_ICD_DATA_LOCATION = f"{SERVER}/input/raw_data/DIAGNOSES_ICD.csv"
    PROCEDURES_ICD_DATA_LOCATION = f"{SERVER}/input/raw_data/PROCEDURES_ICD.csv"
    WORD2VEC_EMBEDDING_LOCATION = f"{SERVER}/intermediate/word2vec_embedding"
    EMBEDDING_SIZE = args.embedding_size
    BATCH_SIZE = args.batch_size
    USE_CUDA = args.gpu
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    MAX_LENGTH_OF_SENTENCE = args.max_length_of_sentence
    RUN_DATA_PROCESSING = args.data_processing
    PATIENCE = args.patience
    TOPK = args.top_k
    PREDICTION_SAMPLES = args.prediction_samples
    __train__ = args.train
    device = torch.device(
        "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if RUN_DATA_PROCESSING:
        logger.info("Starting Data Processing")
        data_processing()
    seqs_file, labels_file = sequencer(generate_dataset_splits)
    if __train__:
        logger.info("Starting Model Training")
        new_run(train_model, seqs_file, labels_file, topk=TOPK)
    train_losses_cnn, valid_losses_cnn, train_macro_auc_rocs_cnn,\
        valid_macro_auc_rocs_cnn, test_results_cnn, train_losses_gru, valid_losses_gru, \
        train_macro_auc_rocs_gru, valid_macro_auc_rocs_gru, test_results_gru, train_losses_lstm,\
        valid_losses_lstm, train_macro_auc_rocs_lstm, valid_macro_auc_rocs_lstm, test_results_lstm = restore_losses()
    with open(REVERSE_ICD9_INDEX_DICT_FILE, 'rb') as file_name:
        reverse_icd_index_dict = pickle.load(file_name)
    with open(REVERSE_VOCAB_DICT_FILE, 'rb') as file_name:
        reverse_vocab_dict = pickle.load(file_name)
    with open(seqs_file[-1], 'rb') as file_name:
        sample_test = pickle.load(file_name)

    top_k_predictions(logger, PREDICTION_SAMPLES, test_results_cnn, test_results_gru,
                      test_results_lstm,
                      reverse_icd_index_dict,
                      reverse_vocab_dict, sample_test)
    charts(train_losses_cnn, valid_losses_cnn,
           train_macro_auc_rocs_cnn, valid_macro_auc_rocs_cnn,
           train_losses_gru, valid_losses_gru, train_macro_auc_rocs_gru,
           valid_macro_auc_rocs_gru, train_losses_lstm, valid_losses_lstm,
           train_macro_auc_rocs_lstm, valid_macro_auc_rocs_lstm)
