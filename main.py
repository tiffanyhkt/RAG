from RAG import process_single_query
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
        kg_manager.create_constraints()
        print("Create knowledge graph")
        result = process_single_query("", pdf_path)  
        if result and result.get("contexts"):
            chunks = [{"content": ctx, "metadata": {"source": pdf_path}} for ctx in result["contexts"]]
            kg_manager.create_knowledge_graph(chunks)
            print("Finished create knowledge graph")
        else:
            print("Please check pdf.")
            # return

        query = input("User Query: ").strip()
        
        if query:
            # RAG
            result = process_single_query(query, pdf_path)
            if result:
                print("RAG Response：", result["content"])
                if result["contexts"]:
                    print("Referenced contexts：")
                    for ctx in result["contexts"]:
                        print("-", ctx)
            else:
                print("RAG Error")
        else:
            # 使用 KG 生成問題
            questions = kg_manager.generate_questions(num_questions=5)
            if questions:
                print("Generated Questions：")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
            else:
                print("Questions Generation Error")
                
    except Exception as e:
        print(f"Error：{str(e)}")
    finally:
        kg_manager.close()

if __name__ == "__main__":
    pdf_path = "2408.09869v5.pdf"
    main(pdf_path) 