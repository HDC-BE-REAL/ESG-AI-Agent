
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (API KEY)
load_dotenv()

VECTOR_DB_DIR = "vector_db/esg_all"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def test_rag_generation():
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"❌ Directory not found: {VECTOR_DB_DIR}")
        return

    print("🔌 Loading Vector DB & LLM...")
    
    # 1. Setup Retrieval
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma(
        collection_name="esg_all",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. Setup LLM & Prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    template = """
    당신은 ESG 전문가 AI입니다. 아래 제공된 [Context]를 바탕으로 질문에 답변해주세요.
    반드시 **한국어**로 답변해야 합니다.
    
    [Context]
    {context}
    
    Question: {question}
    
    Answer (in Korean):
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Build Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Run Test
    query = "최근 규제 동향은?"
    print(f"\n❓ 질문: '{query}'")
    print("🤖 답변 생성 중 (LLM 호출)...")
    
    try:
        response = rag_chain.invoke(query)
        print("\n" + "="*50)
        print(response)
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_rag_generation()
