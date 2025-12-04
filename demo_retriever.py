
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from config import PipelineConfig
from esg_rag_pipeline import ESGRAGPipeline
from advanced_retriever import AdvancedRetriever

PDF_PATH = Path("pdf_folder/포스코건설_ESG보고서.pdf")

def main():
    # 1. Configuration
    load_dotenv() # Added load_dotenv call
    config = PipelineConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY")
    )
    
    # API Key Check
    if not config.llama_cloud_api_key or config.llama_cloud_api_key == "your_llama_cloud_api_key":
        print("Error: LLAMA_CLOUD_API_KEY is not set in config.py or environment variables.")
    
    # 2. Re-ingestion
    print("\n=== 1. Ingestion Step ===")
    reingest = "--reingest" in sys.argv
    
    if reingest:
        print("Re-ingestion requested via command line.")
        pdf_path = PDF_PATH
        
        # Check file existence
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            if os.path.exists("pdf_folder"):
                pdfs = [f for f in os.listdir("pdf_folder") if f.endswith(".pdf")]
                if pdfs:
                    pdf_path = os.path.join("pdf_folder", pdfs[0])
                    print(f"Using found PDF: {pdf_path}")
        
        if os.path.exists(pdf_path):
            print(f"Processing {pdf_path}...")
            pipeline = ESGRAGPipeline(
                llama_cloud_api_key=config.llama_cloud_api_key,
                embedding_model=config.embedding_model,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                persist_directory=config.persist_directory
            )
            # Force re-parse/re-process to get metadata
            pipeline.process_pdf(pdf_path, use_cache=False) 
        else:
            print("No PDF found to ingest.")
    else:
        print("Skipping re-ingestion. Use --reingest to force update.")

    # 3. Retrieval Demo
    print("\n=== 2. Retrieval Step ===")
    try:
        retriever = AdvancedRetriever(config)
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        print("Make sure you have ingested documents at least once.")
        return

    query = "탄소 중립 전략"
    
    # A. Similarity Search
    print(f"\n[A] Similarity Search for '{query}'")
    results = retriever.retrieve(query, k=3)
    for i, doc in enumerate(results):
        print(f"{i+1}. (Length: {len(doc.page_content)}) {doc.page_content}\n   Meta: {doc.metadata}\n")

    # B. MMR Search
    print(f"\n[B] MMR Search for '{query}'")
    results = retriever.retrieve(query, k=3, search_type="mmr")
    for i, doc in enumerate(results):
        print(f"{i+1}. (Length: {len(doc.page_content)}) {doc.page_content}\n   Meta: {doc.metadata}\n")

    # C. Metadata Filter
    print(f"\n[C] Filtered Search (Page <= 10)")
    results = retriever.retrieve(query, k=3, filter_page_max=10)
    for i, doc in enumerate(results):
        print(f"{i+1}. (Length: {len(doc.page_content)}) {doc.page_content}\n   Meta: {doc.metadata}\n")

    # D. Query Rewriting
    if config.openai_api_key:
        print(f"\n[D] Query Rewriting Search")
        try:
            results = retriever.retrieve(query, k=3, use_rewrite=True)
            for i, doc in enumerate(results):
                print(f"{i+1}. (Length: {len(doc.page_content)}) {doc.page_content}\n   Meta: {doc.metadata}\n")
        except Exception as e:
            print(f"Query rewriting failed: {e}")
    else:
        print("\n[D] Query Rewriting skipped (No OPENAI_API_KEY)")

if __name__ == "__main__":
    main()
