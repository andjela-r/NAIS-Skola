from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection, utility, connections, db
from sentence_transformers import SentenceTransformer
import csv
import pandas as pd
from tqdm import tqdm
import time

connections.connect(host='localhost', port='19530')


#database = db.create_database("skola")
db.using_database("skola")
client = MilvusClient(uri= "http://localhost:19530", db_name="skola")

#Definisanje parametara
DIMENSION = 384
COLLECTION_NAME = "izvestaji"
MAX_ROWS = 100
COUNT = 100
BATCH_SIZE = 128

#Brisanje kolekcije ako vec postoji
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='razred', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='redni_br_casa', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='tekst', dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name='subj_ocena', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='ocena_standarda', dtype=DataType.VARCHAR, max_length=50),  
    FieldSchema(name='tekst_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)         
]

schema = CollectionSchema(fields=fields)

collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.create_index(field_name="tekst_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
collection.load()
print('Collection created and indices created')

transformer = SentenceTransformer('all-MiniLM-L6-v2')

def csv_load(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row
        for row in reader:
            if '' in (row[0], row[1], row[2], row[3], row[4]):
                continue
            yield (row[0], row[1], row[2], row[3], row[4])
    
def embed_insert(data):
    tekst_emb = transformer.encode(data[2]) 
    ins = [
       
        data[0], data[1], data[2], data[3], data[4],
        [x for x in tekst_emb],
    ]
    collection.insert(ins)

def embed_search(data):
    embeds = transformer.encode(data)
    return [x for x in embeds]


count = 0
data_batch = [[], [], [], [], []]

try:
    for col0, col1, col2, col3, col4 in csv_load("data_csv/output2.csv"):
        if count <= COUNT:
            data_batch[0].append(col0)
            data_batch[1].append(col1)
            data_batch[2].append(col2)
            data_batch[3].append(col3)
            data_batch[4].append(col4)
            
            if len(data_batch[0]) % BATCH_SIZE == 0:
                embed_insert(data_batch)
                data_batch = [[], [], [], [], []]
            count += 1
        else:
            break

    if len(data_batch[0]) != 0:
        embed_insert(data_batch)
        count += len(data_batch[0])

    collection.flush()
    print('Inserted data successfully')
    print('Number of inserted items:', count)
except Exception as e:
    print('Error occurred during data insertion:', str(e))
    raise e