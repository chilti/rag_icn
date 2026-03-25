import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "../")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(name="ICN_InCitesRecordsMicroTopics")

results = collection.query(query_texts="black holes", n_results=5)

print(results)
