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
COLLECTION_NAME = "pripreme_za_cas"
MAX_ROWS = 106
COUNT = 106
BATCH_SIZE = 128

#Brisanje kolekcije ako vec postoji
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='nastavna_jedinica', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='razred', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='redni_br_casa', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='tip_nastavnog_casa', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='glavni_deo', dtype=DataType.VARCHAR, max_length=2000),  
    FieldSchema(name='domaci', dtype=DataType.VARCHAR, max_length=2001),
    FieldSchema(name='glavni_deo_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)         
]

schema = CollectionSchema(fields=fields)

collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.create_index(field_name="glavni_deo_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
collection.load()
print('Collection created and indices created')

transformer = SentenceTransformer('all-MiniLM-L6-v2')

def csv_load(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row
        for row in reader:
            if '' in (row[0], row[1], row[2], row[3], row[4], row[5]):
                continue
            yield (row[0], row[1], row[2], row[3], row[4], row[5])
    
def embed_insert(data):
    glavni_deo_emb = transformer.encode(data[4]) 
    ins = [
       
        data[0], data[1], data[2], data[3], data[4], data[5],
        [x for x in glavni_deo_emb],
    ]
    collection.insert(ins)

def embed_search(data):
    embeds = transformer.encode(data)
    return [x for x in embeds]


count = 0
data_batch = [[], [], [], [], [], []]

try:
    for col0, col1, col2, col3, col4, col5 in csv_load("data_csv/output3.csv"):
        if count <= COUNT:
            data_batch[0].append(col0)
            data_batch[1].append(col1)
            data_batch[2].append(col2)
            data_batch[3].append(col3)
            data_batch[4].append(col4)
            data_batch[5].append(col5)
            
            if len(data_batch[0]) % BATCH_SIZE == 0:
                embed_insert(data_batch)
                data_batch = [[], [], [], [], [], []]
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

def write_ouput(res, start, end):
    for hits_i, hits in enumerate(res):
        print('Naslov nastavne jedinice:', search_terms[hits_i])
        print('Vreme pretrage:', end-start)
        print('Rezultat:')
        for hit in hits:
            print( hit['entity'].get('nastavna_jedinica'), '----', hit['distance'])
        print()

print("Ucitavanje kolekcije")

client.load_collection(
    collection_name = COLLECTION_NAME
)

res = client.describe_collection(
    collection_name= COLLECTION_NAME
)

print("Upit 1")

search_terms = [
    'IKT', 
    'Rad',
    'programska struktura'
]

query_vectors = embed_search(search_terms)
start = time.time()

res = client.search(
    collection_name = COLLECTION_NAME,
    data = query_vectors,
    anns_field= "glavni_deo_emb",
    limit = 1,
    output_fields=['id', 'nastavna_jedinica', 'redni_br_casa', 'glavni_deo']
)

end = time.time()
write_ouput(res, start, end)