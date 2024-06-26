import csv
from fastapi import FastAPI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient, db
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import Query
from fastapi.responses import JSONResponse
from typing import List, Dict
import logging
import uvicorn

app = FastAPI()

DIMENSION = 384
COLLECTION_NAME = "pripreme_za_cas"
COLLECTION_NAME2 = "izvestaji"
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
MAX_ROWS = 100
COUNT = 100
BATCH_SIZE = 128

@app.get("/")
def hello_world():
    return {"message": "OK"}

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
db.using_database("skola")
client = MilvusClient(uri="http://127.0.0.1:19530" , db_name="skola")
collection = Collection(name=COLLECTION_NAME)

# Check connection to Milvus and count items inserted into collection
@app.get("/test-milvus-connection/")
async def test_milvus_connection():
    try:
        # Check if connected to Milvus
        status = client.get_collection_stats(collection_name="pripreme_za_cas")
        #res = client.describe_collection(collection_name="pripreme_za_cas")
        #print(res)
        return {"message": "Connected to Milvus", "status": status}
    except Exception as e:
        return {"message": "Error occurred during Milvus connection:", "error": str(e)}
    
transformer = SentenceTransformer('all-MiniLM-L6-v2')
# Insert
@app.post("/api/collections/{collection_name}/insert_data") 
async def insert_data(collection_name: str):
    try:
        collection = Collection(name=collection_name)
        if collection_name=="pripreme_za_cas":
            # Example data
            nastavna_jedinica = "Example Unit"
            razred = "Example Grade"
            redni_br_casa = "Example Class Number"
            tip_nastavnog_casa = "Example Class Type"
            glavni_deo = "This is the main part of the lesson."
            domaci = "This is the homework."

            data = [
                [nastavna_jedinica],
                [razred],
                [redni_br_casa],
                [tip_nastavnog_casa],
                [glavni_deo],
                [domaci]
            ]

            glavni_deo_emb = transformer.encode(data[4]) 
            ins = [data[0], data[1], data[2], data[3], data[4], data[5],[x for x in glavni_deo_emb]]
        else:
            razred = "Prvi razred"
            redni_br_casa = "15"
            tekst = "Aktivnost je odlicna bila"
            subj_ocena = "10"
            ocena_standarda = "10"

            data = [
                [razred],
                [redni_br_casa],
                [tekst],
                [subj_ocena],
                [ocena_standarda]
            ]

            tekst_emb = transformer.encode(data[2]) 
            ins = [data[0], data[1], data[2], data[3], data[4],[x for x in tekst_emb]]
        collection.insert(ins)
        collection.flush()
        
        return {"message": "Inserted entity:", "data": data}
    except Exception as e:
        return {"message": "Error occurred:", "error": str(e)}
    
#Upsert not supported for autoIds

# Delete    
@app.delete("/api/collections/{collection_name}/delete_data") 
async def delete_data(collection_name: str):
    try:
        collection = Collection(name=collection_name)
        if collection_name=="pripreme_za_cas":
            res = client.delete(
            collection_name="pripreme_za_cas",
            filter="nastavna_jedinica LIKE 'Rad%'"
        )
            print(res)

        else:
            res = client.delete(
            collection_name="izvestaji",
            filter="subj_ocena LIKE '7'"
        )
            print(res)
        
        return {"message": "Succesfully deleted entities"}
    except Exception as e:
        return {"message": "Error occurred:", "error": str(e)}

# Load SentenceTransformer model
client = MilvusClient(uri= "http://localhost:19530", db_name="skola")

@app.get("/api/collections/{collection_name}/getvector1/{vector_id}") #radi
#/api/collections/pripreme_za_cas/getvector1/450729182199022372
#/api/collections/izvestaji/getvector1/450729182199022400
async def get_vector(
    collection_name: str,
    vector_id: int
):
    try:
        client = MilvusClient(uri= "http://localhost:19530", db_name="skola")

        vector_data = client.get(collection_name=collection_name, ids=[vector_id])

        if vector_data:
            if collection_name=="pripreme_za_cas":
                vector_dict = {
                    "id": vector_id,
                    "nastavna_jedinica": vector_data[0]["nastavna_jedinica"],
                    "redni_br_casa": vector_data[0]["redni_br_casa"],
                    "glavni_deo": vector_data[0]["glavni_deo"]
                }
            else:
                vector_dict = {
                    "id": vector_id,
                    "redni_br_casa": vector_data[0]["redni_br_casa"],
                    "razred": vector_data[0]["razred"],
                    "tekst": vector_data[0]["tekst"]
                }
            return JSONResponse(content={"vector_data": vector_dict})
        else:
            return JSONResponse(content={"message": "Vector not found"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/collections/{collection_name}/simple-query-1")
async def query_collection1(collection_name: str, filter: str = Query('Rad%', title="Filter", description="Filter query using LIKE operator"), 
                           limit: int = Query(3, title="Limit", description="Number of results to return")):
    try:
        client = MilvusClient(uri= "http://localhost:19530", db_name="skola")
        res = client.query(
            collection_name=collection_name,
            filter=f"nastavna_jedinica LIKE '{filter}'",
            output_fields=["nastavna_jedinica", "glavni_deo"],
            limit=limit
        )
        return {"query_result": res}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/collections/{collection_name}/simple-query-2")
async def query_collection2(collection_name: str, filter: str = Query('Projektni zadatak', title="Filter", description="Filter query using LIKE operator"), 
                           limit: int = Query(3, title="Limit", description="Number of results to return")):
    try:
        client = MilvusClient(uri= "http://localhost:19530", db_name="skola")
        res = client.query(
            collection_name=collection_name,
            filter=f"nastavna_jedinica LIKE '{filter}'",
            output_fields=["nastavna_jedinica", "glavni_deo"],
            limit=limit
        )
        return {"query_result": res}
    except Exception as e:
        return {"error": str(e)}

model = SentenceTransformer('all-MiniLM-L6-v2')

def search_with_embedding(search_term: str) -> List[Dict[str, str]]:
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("Generating embedding for the search term")
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)
        logging.info(f"Generated embeddingg: {embedding}")

        search_params = {
            "metric_type": "L2"
        }
        logging.info("Creating a search request for Milvus")
        client = MilvusClient(uri= "http://localhost:19530", db_name="skola")
        results = client.search(
            collection_name='pripreme_za_cas',
            data=embedding,
            anns_field= "glavni_deo_emb",
            search_params=search_params, 
            limit=20,
            output_fields=['nastavna_jedinica', 'redni_br_casa', 'glavni_deo']
        )
   
        search_results = []
        for hit in results[0]:
            nastavna_jedinica = hit.get('nastavna_jedinica')
            redni_br_casa = hit.get('redni_br_casa')
            glavni_deo = hit.get('glavni_deo')
            search_results.append({"nastavna_jedinica": nastavna_jedinica, "redni_br_casa": redni_br_casa, "glavni_deo": glavni_deo})
        
        logging.info(f"Search results from method: {results}")
        return results
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return [{"error": str(e)}]

@app.get("/api/search/") #polu-radi
async def perform_search(search_term: str = Query(..., title="Search Term", description="Term to search for")):
    search_results = search_with_embedding(search_term)
    print("Received search term:", search_term)
    print("Search results from endpoint:", search_results)

    return {"search_results": search_results}

@app.get("/api/collections/{collection_name}/vector_search_with_filter_1/")
async def vector_search_with_filter1(collection_name:str, search_term: str, razred: str, tip_nastavnog_casa: str):
    try:
        collection = Collection(name=collection_name)
        embedding = model.encode([search_term])

        # Definisanje uslova filtriranja
        filter_query = f"razred == '{razred}' && tip_nastavnog_casa == '{tip_nastavnog_casa}'"
        
        # Pretraga u Milvus kolekciji
        search_params = {
            "metric_type": "L2"
        }
        results = collection.search(
            data=embedding,
            anns_field="glavni_deo_emb",
            param=search_params,
            limit=10,
            output_fields=['nastavna_jedinica', 'razred', 'redni_br_casa', 'tip_nastavnog_casa'],
            expr=filter_query
        )

        search_results = []
        for hit in results[0]:
            search_results.append(hit)

        return search_results
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/collections/{collection_name}/vector_search_with_filter_2/")
async def vector_search_with_filter2(collection_name:str, search_term: str, razred: str, ocena_standarda: str):
    try:
        collection = Collection(name=collection_name)
        embedding = model.encode([search_term])

        # Definisanje uslova filtriranja
        filter_query = f"razred == '{razred}' && ocena_standarda IN({ocena_standarda})"
        
        # Pretraga u Milvus kolekciji
        search_params = {
            "metric_type": "L2"
        }
        results = collection.search(
            data=embedding,
            anns_field="glavni_deo_emb",
            param=search_params,
            limit=10,
            output_fields=['nastavna_jedinica', 'razred', 'redni_br_casa', 'tip_nastavnog_casa'],
            expr=filter_query
        )

        search_results = []
        for hit in results[0]:
            search_results.append(hit)

        return search_results
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/collections/{collection_name}/vector_search_with_filter_iterator/")
async def vector_search_with_filter_iterator(collection_name:str, razred: str):
    try:
        collection = Collection(name=collection_name)
        expr = f"razred == '{razred}'"
        output_fields = ['nastavna_jedinica', 'razred', 'redni_br_casa', 'glavni_deo']

        batch_size = 10
        limit = 100

        query_iterator = collection.query_iterator(batch_size, limit, expr, output_fields)

        while True:
            res = query_iterator.next()
            if len(res) == 0:
                print("query iteration finished, close")
                query_iterator.close()
                break
            for i in range(len(res)):
                print(res[i])
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/collections/{collection_name}/hybrid_search/")
async def hybrid_search(collection_name:str, search_term: str, razred: str, redni_br_casa: str):
    try:
        # Generisanje embeddinga za search_term
        collection = Collection(name=collection_name)
        embedding = model.encode([search_term])

        # Definisanje uslova filtriranja
        filter_query = f"razred == '{razred}' || redni_br_casa == '{redni_br_casa}'"
        
        # Pretraga u Milvus kolekciji
        search_params = {"metric_type": "L2"}
        results = collection.search(
            data=embedding,
            anns_field="glavni_deo_emb",
            param=search_params,
            limit=10,
            output_fields=['nastavna_jedinica', 'razred', 'redni_br_casa'],
            expr=filter_query
        )
        search_results = []
        for hit in results[0]:
            search_results.append(hit)

        return search_results
    except Exception as e:
        return {"error": str(e)}
    

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Example endpoint that generates a PDF report
@app.get("/api/generate-pdf-report")
async def generate_pdf_report():
    try:
        query_result_1 = await query_collection1("pripreme_za_cas", "Rad%", 3)
        query_result_2 = await query_collection2("pripreme_za_cas", "%Projektni zadatak%", 3)
        vector_search_result_1 = await vector_search_with_filter1("pripreme_za_cas", "Paint", "VI", "obrada")
        vector_search_result_2 = await vector_search_with_filter2("izvestaji", "algoritmi", "5", "8,9")

        filename = "report.pdf"
        doc = SimpleDocTemplate(filename)
        story = []
        styles = getSampleStyleSheet()
        query_results_style = TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Align text to top
                ('WORDWRAP', (0, 0), (-1, -1), True)  # Enable word wrap
            ])
        def create_wrapped_table_data(results):
            table_data = []
            for result in results:
                row = [Paragraph(result['nastavna_jedinica'], styles['Normal']), Paragraph(result['glavni_deo'], styles['Normal'])]
                table_data.append(row)
            return table_data
        
        def create_wrapped_table_data2(results, keys):
            table_data = []
            for result in results:
                row = [Paragraph(result[key], styles['Normal']) for key in keys]
                table_data.append(row)
            return table_data

        def create_wrapped_vector_table_data(results):
            table_data = []
            for result in results:
                row = [
                    Paragraph(result['nastavna_jedinica'], styles['Normal']),
                    Paragraph(result['tip_nastavnog_casa'], styles['Normal']),
                    Paragraph(result['razred'], styles['Normal']),
                    Paragraph(result['glavni_deo'], styles['Normal'])
                ]
                table_data.append(row)
            return table_data
        
        #QUERY 1
        query_results_table = create_wrapped_table_data(query_result_1['query_result'])
        table = Table(query_results_table, colWidths=[200, 200])
        table.setStyle(query_results_style)
        story.append(Paragraph("Nastavne jedinice koje u naslovu imaju rec 'Rad'", styles['Heading1']))
        story.append(table)
        story.append(Paragraph("<br/><br/>", styles['Normal']))
        
        #QUERY 2
        query_results_table = create_wrapped_table_data(query_result_2['query_result'])
        table = Table(query_results_table, colWidths=[200, 200])
        table.setStyle(query_results_style)
        story.append(Paragraph("Nastavne jedinice koje u naslovu imaju rec 'Projektni zadatak'", styles['Heading1']))
        story.append(table)
        story.append(Paragraph("<br/><br/>", styles['Normal']))

        #VECTOR SEARCH 1
        
        if isinstance(vector_search_result_1, list):
            vector_search_table = create_wrapped_table_data2(
                vector_search_result_1,['nastavna_jedinica', 'tip_nastavnog_casa', 'razred', 'glavni_deo']
            )
            table = Table(vector_search_table, colWidths=[100, 100, 100, 100])
            table.setStyle(query_results_style)
            story.append(Paragraph("pripreme_za_cas, Paint, VI, obrada", styles['Heading1']))
            story.append(table)
            story.append(Paragraph("<br/><br/>", styles['Normal']))

        # Build the PDF document
        doc.build(story)

        return {"message": "PDF report generated successfully", "filename": filename}
    except Exception as e:
        return {"error": str(e)}
