### Steps ####
1) Download MIMIC III data from physionet.org 
2) Unzip the files and extract the csv
3) Start Hadoop cluster \
3.1 Switch to hdoop user su - hdoop \
3.2 ssh localhost \
3.3 Start services \
cd hadoop-3.2.1/sbin/  
./start-dfs.sh \
./start-yarn.sh \
3.4 Check jps for running services \
jps
```
hdoop@pop-os:~/hadoop-3.2.1/sbin$ jps
54400 DataNode
54289 NameNode
54599 SecondaryNameNode
55287 Jps
54907 NodeManager
54812 ResourceManager
```
4) Move the files to hadoop
4.1 (hw5) hdoop@pop-os:~$ cd /home/ubuntu/Documents/CAML_GroupProject/physionet.org/files/mimiciii/1.4 \
4.2 hdfs dfs -put NOTEEVENTS.csv /input/raw_data/
4.3 hdfs dfs -put DIAGNOSES_ICD.csv /input/raw_data/ \
4.5 hdfs dfs -put PROCEDURES_ICD.csv /input/raw_data \
4.6 Check content on disk - hdfs dfs -ls /input/raw_data/
```
(hw5) hdoop@pop-os:/home/ubuntu/Documents/CAML_GroupProject/physionet.org/files/mimiciii/1.4$ hdfs dfs -ls /input/raw_data/
Found 2 items
-rw-r--r--   1 hdoop supergroup   12548562 2021-04-09 16:19 /input/raw_data/NOTEEVENTS.csv
-rw-r--r--   1 hdoop supergroup   19137527 2021-04-09 16:30 /input/raw_data/DIAGNOSES_ICD.csv
```
5) Activate conda environment - source anaconda3/bin/activate
6) Activate pyspark environment - source activate hw5
7) From host, give all write permissions to python source code folder
```
chmod 777 src
```
8) sudo chmod 777 main.py
8.1) ./main.py -h
9) Usage 
```
usage: main.py [-h] [--server [SERVER]] [--data_processing] [--gpu] [--train]
               [--embedding_size [EMBEDDING_SIZE]] [--batch_size [BATCH_SIZE]]
               [--num_workers [NUM_WORKERS]] [--num_epochs [NUM_EPOCHS]]
               [--patience [PATIENCE]]
               [--max_length_of_sentence [MAX_LENGTH_OF_SENTENCE]]
               [--top_k [TOP_K]] [--prediction_samples [PREDICTION_SAMPLES]]

Final Group Project

optional arguments:
  -h, --help            show this help message and exit
  --server [SERVER]     Provide address of server (default:
                        hdfs://localhost:9000)
  --data_processing     Data Preprocessing (default: False)
  --gpu                 Use GPU for training (default: False)
  --train               Train Model (default: False)
  --embedding_size [EMBEDDING_SIZE]
                        Size of embedding layer (default: 100)
  --batch_size [BATCH_SIZE]
                        Batch size for training the model (default: 32)
  --num_workers [NUM_WORKERS]
                        Number of Available CPUs (default: 5)
  --num_epochs [NUM_EPOCHS]
                        Number of Epochs for training the model (default: 10)
  --patience [PATIENCE]
                        Number of epochs Early Stopping (default: 2)
  --max_length_of_sentence [MAX_LENGTH_OF_SENTENCE]
                        Maximum length of sentence for Spark Word2Vec Model
                        (default: 1000)
  --top_k [TOP_K]       Top k predictions (default: 5)
  --prediction_samples [PREDICTION_SAMPLES]
                        Number of prediction samples (default: 5)
```
10) ./main.py --data_processing --gpu --train
