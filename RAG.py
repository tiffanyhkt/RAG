from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os
from datetime import datetime
from chromadb.config import Settings
from config import OPENAI_API_KEY

#OpenAI API
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#Embedding and llm config
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
chat_client = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0
)

def process_query(query, pdf_path, chat_history):
    try:
        #PDF name 作為 DB name 存在 ./chroma_db 路徑下
        db_name = os.path.splitext(os.path.basename(pdf_path))[0]
        persist_directory = f"./chroma_db/{db_name}"
        
        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

        #Chroma local DB
        vector_store = Chroma(
            collection_name=db_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            client_settings=chroma_settings
        )

        if vector_store._collection.count() == 0: #If DB 是空的
            
            # Docling PDF解析，切成chunks，預設依照 token 限制切分
            # Docling layout chunking + tokenizer max tokens (sentence-transformers/all-MiniLM-L6-v2 model) -> 512 tokens
            loader = DoclingLoader(
                file_path=pdf_path,
                export_type=ExportType.DOC_CHUNKS,
                                   )
            docs = loader.load()
            
            all_chunks = []
            
            #chunk's content & metadata(source, page)
            for doc in docs:
                chunk = doc.page_content #Finished Docling chunking's chunk content                
                metadata = {
                        "source": doc.metadata["source"], #pdf source
                        "page": doc.metadata["dl_meta"]["doc_items"][0]["prov"][0]["page_no"] #page no
                    }
                all_chunks.append({
                    "content": chunk,
                    "metadata": metadata
                    })
            
            #Run more texts through the embeddings and add to the vectorstore (texts, metadatas)
            vector_store.add_texts(
                texts=[chunk["content"] for chunk in all_chunks],
                metadatas=[chunk["metadata"] for chunk in all_chunks]
            )
            print(f"Total added {len(all_chunks)} chunks")
        else:
            print("Indexed before, using existing vectors")

        #如果DB不是空的，可以直接檢索
        #vector search
        retriever_mmr = vector_store.as_retriever(
            search_type="mmr", #Maximal Marginal Relevance 演算法，平衡相關性和多樣性
            search_kwargs={
                "k": 3,
                "fetch_k": 20, #候選chunks中去選
                "lambda_mult": 0.7 #0.7 查詢相似度，0.3 看候選 k 間的差異度
            }
        )
        #Distance: cosine similarity or 內積
        #已選的東西和未選的distance(差異度)

        #keyword search
        #Chroma 撈出 chunk text -> Document (all docs) 來檢索
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

        # Hybrid: MMR(vector) + BM25(keyword) 先各取出3筆結果->結果合併->根據相關性分數與加權排序
        # Chunk 總分 = BM25_score × 0.5 + MMR_score × 0.5 => (X) score 要先normalized 在做 a & b 係數調配  or 使用RRF
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever_mmr],
            weights=[0.5, 0.5]  
        )
        #建議用RRF或是自己去調配 a & b

        # 加權總分，排序 & 挑前 3 chunk
        retrieved_docs = hybrid_retriever.invoke(query)[:3]
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generation
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        response = chat_client.invoke([
            {
                "role": "system",
                "content": f'''You are an intelligent assistant operating in a Retrieval-Augmented Generation (RAG) setting. 
                Your task is to answer user queries strictly based on the provided context from the SELECTED FILES.
                '''},
            {
                "role": "system",
                "content": f'''
                Guidelines:
                1. Use only the information retrieved from the SELECTED FILES. Do not rely on your own prior knowledge or external information.
                2. If the provided content does not contain sufficient information to answer the question, politely inform the user that no relevant information was found in the source. 
                Then, you may offer a general answer from your own perspective, clearly indicating it is outside the source.
                3. If the user's request is unclear or ambiguous, ask for clarification before answering.
                4. Always reply in the same language the user used most recently.
                5. Ensure that your responses are concise, accurate, and grounded in the retrieved content.
                
                Current timestamp:{current_time}'''},
            {
                "role": "system",
                "content": f'''Retrieved Context (SELECTED FILES):\n {contexts}'''},] 
                + chat_history + [
            {
                "role": "user", 
                "content": query
            }
        ])

        return {
            "type": "answer",
            "content": response.content,
            "Referenced contexts": contexts
        }
        
    except Exception as e:
        print(f"Error message：{str(e)}")
        return None


if __name__ == "__main__":
    # query = "Docling 是如何辨識 table 的結構?" 
    pdf_path = "2408.09869v5.pdf"
    chat_history = []
    
    print("Conversation starts:")
    while True:
        query = input("User Query: ")
        if query.lower() in ["exit", "quit", "q"]:
                print("Conversation ends")
                break
        
        result = process_query(query, pdf_path, chat_history)
        if result:
            print("Response: ", result['content'])
            print("Referenced Contexts: ", result['Referenced contexts'])
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": result['content']})
        else:
            print("Failed to generate response, please try again.")


    #What is Table 1 about?
    #跨頁內容 (有抓到contexts)
    #Ensemble and rerank 怎麼選? 為什麼不選rerank?
