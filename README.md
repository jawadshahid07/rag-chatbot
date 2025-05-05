# Automobile RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers automobile-related questions using local data and a locally running LLM via Ollama. It supports questions related to maintenance, troubleshooting, specifications, and more.

## Features

- Uses Hugging Face embeddings with FAISS for semantic search
- Loads and processes structured JSON data
- Uses Ollama to run a local LLM (e.g., LLaMA 2, Mistral)
- Built with LangChain for modularity and scalability

## Tech Stack

- LangChain
- Ollama (local LLM execution)
- Sentence Transformers (Hugging Face)
- FAISS (vector storage)
- Python 3.10+

## Project Structure

```
rag-chatbot/
├── data/                 # JSON files (qa_data.json, car_specs.json)
├── rag/
│   ├── data_loader.py    # Load and chunk documents
│   ├── embedder.py       # Embedding and vector store creation
│   ├── retriever.py      # FAISS-based retrieval logic
│   ├── generator.py      # LLM wrapper using Ollama
│   └── rag_pipeline.py   # Pipeline orchestration
├── main.py               # CLI entrypoint
├── .env                  # Contains OLLAMA_MODEL
├── requirements.txt
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Set up a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Create a `.env` file:
```env
OLLAMA_MODEL=llama2
```

4. Install Ollama and pull the model:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama2
```

5. Add your data to the `data/` folder as `qa_data.json` and `car_specs.json`

## Run

```bash
python main.py
```

## Next Steps

- Add scraping to expand data coverage
- Integrate a web interface
- Extend to agent-based architecture