from langchain_docling import DoclingLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os
from datetime import datetime
from chromadb.config import Settings

#OpenAI API
os.environ["OPENAI_API_KEY"] = "" #Filled in key 

#Embedding and llm config
embeddings = OpenAIEmbeddings()
chat_client = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0
)

def process_single_query(query, pdf_path):
    try:
        db_name = os.path.splitext(os.path.basename(pdf_path))[0]
        persist_directory = f"./chroma_db/{db_name}"
        
        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

        vector_store = Chroma(
            collection_name=db_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            client_settings=chroma_settings
        )

        if vector_store._collection.count() == 0:
            # PDF解析
            loader = DoclingLoader(file_path=pdf_path)
            docs = loader.load()
            
            # 分割文本
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
            
            all_chunks = []
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    simple_metadata = {
                        "source": pdf_path,
                        "page": getattr(doc.metadata, "page", 1) if hasattr(doc.metadata, "page") else 1
                    }
                    all_chunks.append({
                        "content": chunk,
                        "metadata": simple_metadata
                    })
            
            vector_store.add_texts(
                texts=[chunk["content"] for chunk in all_chunks],
                metadatas=[chunk["metadata"] for chunk in all_chunks]
            )
            print(f"Added {len(all_chunks)} chunks")
        else:
            print("Used existing vectors")

        retriever_mmr = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        all_docs = []
        results = vector_store.get()
        if results and 'documents' in results:
            for i, text in enumerate(results['documents']):
                metadata = results['metadatas'][i] if 'metadatas' in results and i < len(results['metadatas']) else {}
                all_docs.append(Document(
                    page_content=text,
                    metadata=metadata
                ))

        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 3

        # Hybrid: MMR + BM25
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever_mmr],
            weights=[0.5, 0.5]  
        )

        retrieved_docs = hybrid_retriever.invoke(query)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generation
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        response = chat_client.invoke([
            {
                "role": "system",
                "content": f'''You are a chatbot designed to assist users based on the provided context. Please answer their questions accordingly.
                '''},
            {
                "role": "system",
                "content": f'''Instructions:
                1. Don't use your own knowledge. Use the content of the SELECTED FILES to answer the user.
                2. If the content of the SELECTED FILES is not enough, please tell the user you cannot find any relevant information from the source and try to answer from your perspective in the tone of an assistant.
                3. If user's intention is vague, you should ask for clarification.
                4. You should always reply in the most recent language user used.
                Current time:{current_time}'''},
            {
                "role": "system",
                "content": f'''Content of the SELECTED FILES: {contexts}'''},
            {
                "role": "user", 
                "content": query
            }
        ])

        return {
            "type": "answer",
            "content": response.content,
            "contexts": contexts
        }
        
    except Exception as e:
        print(f"Error message：{str(e)}")
        return None


if __name__ == "__main__":
    query = "What is the overall success rate of Human Performance on VisualWebArena?"
    pdf_path = "2401.13649v2.pdf"
    result = process_single_query(query, pdf_path)