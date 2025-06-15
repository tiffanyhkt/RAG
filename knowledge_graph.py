from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from typing import List, Dict
import json

class KnowledgeGraphManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        )

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE")

    def create_knowledge_graph(self, chunks: List[Dict]):
        print("Creating knowledge graph")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"Working on the {i+1} chunk")
                entities_relations = self._extract_entities_relations(chunk["content"])
                print(f"Extracted entity and relations：{json.dumps(entities_relations, ensure_ascii=False, indent=2)}")
                
                if not entities_relations.get("entities") and not entities_relations.get("relations"):
                    print("Failed to extract, try next chunk")
                    continue
                
                with self.driver.session() as session:
                    for entity in entities_relations.get("entities", []):
                        try:
                            query = """
                                MERGE (c:Concept {name: $name})
                                RETURN c
                            """
                            result = session.run(query, name=entity)
                            if result.single():
                                print(f"Success：{entity}")
                            else:
                                print(f"Failed：{entity}")
                        except Exception as e:
                            print(f"Entity creation error：{str(e)}")
                            continue
                    
                    for relation in entities_relations.get("relations", []):
                        try:
                            query = """
                                MATCH (c1:Concept {name: $source})
                                MATCH (c2:Concept {name: $target})
                                MERGE (c1)-[r:RELATES_TO {type: $type}]->(c2)
                                RETURN r
                            """
                            result = session.run(query, 
                                source=relation["source"],
                                target=relation["target"],
                                type=relation["type"]
                            )
                            if result.single():
                                print(f"Success：{relation['source']} -[{relation['type']}]-> {relation['target']}")
                            else:
                                print(f"Failed：{relation['source']} -[{relation['type']}]-> {relation['target']}")
                        except Exception as e:
                            print(f"Error message：{str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Chunks Processing Error：{str(e)}")
                continue
        
        with self.driver.session() as session:
            entity_count = session.run("MATCH (c:Concept) RETURN count(c) as count").single()["count"]
            print(f"Entitiy Counts：{entity_count}")
            
            relation_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count").single()["count"]
            print(f"Relations Counts：{relation_count}")
            
            if entity_count == 0:
                print("No entities")
            if relation_count == 0:
                print("No relations")

    def _extract_entities_relations(self, text: str) -> Dict:
        prompt = PromptTemplate.from_template(
            """Extract key entities and their relationships from the following text:
            
            Text: {text}
            
            Return the result in JSON format with two fields:
            - entities: a list of extracted entities (important concepts, objects, or topics)
            - relations: a list of relationships, where each relationship includes:
              * source: the source entity
              * target: the target entity
              * type: the type of relationship between them
            
            Example format:
            {{
                "entities": ["Entity1", "Entity2", "Entity3"],
                "relations": [
                    {{
                        "source": "Entity1",
                        "target": "Entity2",
                        "type": "RelationshipType"
                    }},
                    {{
                        "source": "Entity2",
                        "target": "Entity3",
                        "type": "AnotherRelationship"
                    }}
                ]
            }}
            
            Guidelines:
            1. Extract only significant entities that represent key concepts
            2. Create meaningful relationships that reflect actual connections
            3. Use clear and specific relationship types
            4. Ensure all entities in relationships exist in the entities list
            5. Return ONLY the JSON object, without any markdown formatting or code blocks
            """
        )

        try:
            response = self.llm.invoke(prompt.format(text=text))
            print(f"\nLLM Raw Response: {response.content}")
            
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                print(f"Cleaned content: {content}")
                
                result = json.loads(content)
                # Validate result format
                if not isinstance(result, dict):
                    print("LLM response is not a valid JSON")
                    return {"entities": [], "relations": []}
                    
                if "entities" not in result or "relations" not in result:
                    print("LLM response missing required fields")
                    return {"entities": [], "relations": []}
                    
                # Validate entities list
                if not isinstance(result["entities"], list):
                    print("entities field is not a list")
                    result["entities"] = []
                    
                # Validate relations list
                if not isinstance(result["relations"], list):
                    print("relations field is not a list")
                    result["relations"] = []
                else:
                    # Validate each relation format
                    valid_relations = []
                    for rel in result["relations"]:
                        if isinstance(rel, dict) and all(k in rel for k in ["source", "target", "type"]):
                            # Ensure source and target exist in entities
                            if rel["source"] in result["entities"] and rel["target"] in result["entities"]:
                                valid_relations.append(rel)
                            else:
                                print(f"Skipping relation with non-existent entities: {rel}")
                        else:
                            print(f"Skipping invalid relation format: {rel}")
                    result["relations"] = valid_relations
                    
                print(f"Processed Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
                return result
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response as JSON: {str(e)}")
                return {"entities": [], "relations": []}
                
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return {"entities": [], "relations": []}

    def generate_questions(self, num_questions: int = 5) -> List[str]:
        print("Start Generating questions...")

        with self.driver.session() as session:
            check_query = """
                MATCH (c:Concept)
                RETURN count(c) as count
            """
            count = session.run(check_query).single()["count"]
            print(f"Entities counts：{count}")
            
            if count == 0:
                print("Empty Knowledge Graph")
                return []
                
            query = """
                MATCH (c1:Concept)-[r:RELATES_TO]->(c2:Concept)
                RETURN c1.name as source, c2.name as target, r.type as relation
                LIMIT 10
            """
            try:
                result = session.run(query).data()
                print(f"Relations counts：{len(result)}")
                if result:
                    print("Relations：")
                    for r in result[:3]:
                        print(f"- {r['source']} -[{r['relation']}]-> {r['target']}")
                
                if not result:
                    print("No Relations")
                    return []
                    
                prompt = PromptTemplate.from_template(
                    """Based on the following knowledge graph relationships, generate {num_questions} relevant questions:
                    
                    Relationships:
                    {relationships}
                    
                    Generate questions that explore these relationships and concepts. The questions should be:
                    1. Clear and specific
                    2. Focused on understanding the relationships between concepts
                    3. Suitable for testing knowledge of the domain
                    
                    Return a list of questions, one per line.
                    """
                )
                
                relationships_text = "\n".join([
                    f"- {r['source']} {r['relation']} {r['target']}"
                    for r in result
                ])
                
                response = self.llm.invoke(prompt.format(
                    num_questions=num_questions,
                    relationships=relationships_text
                ))
                
                questions = [q.strip() for q in response.content.split('\n') if q.strip()]
                print(f"Generated Questions：")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
                    
                return questions
                    
            except Exception as e:
                print(f"Questions Generating Error：{str(e)}")
                return []


if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password" 

    kg_manager = KnowledgeGraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        kg_manager.create_constraints()
        
        chunks = [
            {
                "content": "Sample content",
                "metadata": {"source": "example.pdf", "page": 1}
            }
        ]
        
        kg_manager.create_knowledge_graph(chunks)
        
        questions = kg_manager.generate_questions(num_questions=5)
        print("Generated questions：")
        for q in questions:
            print(f"- {q}")
        
    finally:
        kg_manager.close() 