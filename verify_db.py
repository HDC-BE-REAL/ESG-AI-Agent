
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Define the path used in the tool
VECTOR_DB_DIR = "vector_db/esg_all"

def verify_vector_db():
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"❌ Directory not found: {VECTOR_DB_DIR}")
        return

    print(f"📂 Checking Vector DB at: {VECTOR_DB_DIR}")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vector_db = Chroma(
            collection_name="esg_all",
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        
        count = vector_db._collection.count()
        print(f"✅ Total Documents: {count}")
        
        if count > 0:
            print("\n🔍 Sample Document:")
            results = vector_db.get(limit=1)
            if results and results['ids']:
                print(f"ID: {results['ids'][0]}")
                print(f"Metadata: {results['metadatas'][0]}")
                print(f"Content: {results['documents'][0][:200]}...")
        else:
            print("⚠️ The database is empty.")
            
    except Exception as e:
        print(f"❌ Error loading Vector DB: {e}")

if __name__ == "__main__":
    verify_vector_db()
