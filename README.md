This project combines **Retrieval-Augmented Generation (RAG)** and **Knowledge Graph Construction** using LangChain, Neo4j, and OpenAI models. 
It extracts context from a PDF, builds a knowledge graph into Neo4j, and provides either RAG-based answers or graph-based question generation.

---

## Features

- Extracts contextual chunks from PDF files
- Builds a Neo4j-based knowledge graph using extracted content
- Supports RAG querying via OpenAI LLMs
- Auto-generates questions from the knowledge graph

---

## Requirements

- Python 3.12.11
- Neo4j 
- OpenAI API Key
- Required Python packages (please refer requirements.txt)

---

## Installation

1. **Clone the repository**:
、、、
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
、、、

2. **Install dependencies**
、、、
pip install -r requirements.txt
、、、

3. **Set your environment variables**
在你的 main.py 或 .env 檔案中加入：
、、、
os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
、、、

4. **Update Neo4j credentials in main.py**
、、、
neo4j_password = "<your-neo4j-password>"
、、、
---

## How to Run
、、、
python main.py
、、、

執行後可以選擇：
1. 輸入查詢：系統會使用 RAG 回答你，根據 PDF 內容回答。
2. Enter 不輸入：系統會根據 Neo4j 知識圖譜 產生相關問題。

---

## Notes
1. PDF 檔案路徑目前為：2401.13649v2.pdf，可自行替換成其他文件。
2. 需確認 Neo4j DB 已啟動，並可從 bolt://localhost:7687 存取。
