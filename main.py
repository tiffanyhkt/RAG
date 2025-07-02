from RAG import process_query
from knowledge_graph import KnowledgeGraphManager
from dotenv import load_dotenv
import os
from config import neo4j_uri, neo4j_user, neo4j_password

def main(pdf_path):

    kg_manager = KnowledgeGraphManager(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    try:
        #先建立 KG
        kg_manager.create_constraints()
        print("Create knowledge graph")
        result = process_query("", pdf_path, chat_history)  
        if result and result.get("Referenced contexts"):
            chunks = [{"content": ctx, "metadata": {"source": pdf_path}} for ctx in result["Referenced contexts"]]
            kg_manager.create_knowledge_graph(chunks)
            print("Finished create knowledge graph")
        else:
            print("Please check pdf.")
            return

        chat_history = []
        print("Conversation starts:")
        while True:
            query = input("User Query: ")
            if query.lower() in ["exit", "quit", "q"]:
                    print("Conversation ends")
                    break
            
            if query != None:
                # RAG
                result = process_query(query, pdf_path, chat_history)
                if result:
                    print("RAG Response：", result["content"])
                    # print("Referenced Contexts: ", result['Referenced contexts'])
                    chat_history.append({"role": "user", "content": query})
                    chat_history.append({"role": "assistant", "content": result['content']})
                else:
                    print("RAG Error. Failed to generate response, please try again.")
            else:
                # 使用 KG 生成問題
                questions = kg_manager.generate_questions(num_questions=5)
                if questions:
                    print("Generated Questions：")
                    for i, q in enumerate(questions, 1):
                        print(f"{q}")
                else:
                    print("Questions Generation Error")
                
    except Exception as e:
        print(f"Error：{str(e)}")
    finally:
        kg_manager.close()

if __name__ == "__main__":
    pdf_path = "2408.09869v5.pdf"
    main(pdf_path) 