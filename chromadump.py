import chromadb

chroma_client = chromadb.PersistentClient(path="./chromadb_persist")
collection = chroma_client.get_or_create_collection(name="travel_data")

results = collection.get()
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(doc)
    print(meta)
    print("---")
