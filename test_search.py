
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_DB_DIR = "vector_db/esg_all"

def test_search():
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"❌ Directory not found: {VECTOR_DB_DIR}")
        return

    print("🔌 Loading Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma(
        collection_name="esg_all",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    queries = [
        "최근 규제 동향은???, 무조건 한국어로 대답해!"
    ]
    
    print("\n🔍 Testing Search Queries:\n")
    
    for query in queries:
        print(f"❓ Query: '{query}'")
        try:
            results = vector_db.similarity_search(query, k=2)
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', '?')
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"   {i}. [{source} p.{page}] {content_preview}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_search()
