# 🔍 SRS RAG Engine

A comprehensive **Retrieval-Augmented Generation (RAG)** system for Software Requirements Specification (SRS) documents with advanced search capabilities, local LLM integration, and visual content support.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 🔎 Multiple Search Strategies
- **Dense Search**: Semantic similarity using embeddings (Cosine distance)
- **Sparse Search**: Keyword matching using BM25 algorithm
- **Hybrid Search**: Combines both strategies with two fusion methods:
  - Score-Aware (RRF) - Normalized score fusion (50/50)
  - Weighted - Custom configurable weights

### 🤖 Local LLM Integration
- **Ollama Integration**: Run LLMs locally on-premise
- **Model Selection**: Switch between available models at runtime
- **Private Processing**: No data sent to external APIs

### 📚 Document Processing
- **Intelligent Chunking**: Automatic document splitting with context preservation
- **Metadata Extraction**: Automatic extraction from structured documents
- **Image Integration**: Link and display images associated with chunks
- **Table Processing**: Handle tabular data within documents
- **VLM Support**: Integration with Vision Language Models for image understanding

### 🗂️ Vector Database
- **Qdrant Integration**: Fast vector similarity search
- **Hybrid Vectors**: 
  - Dense vectors (1024-dim) for semantic search
  - Sparse vectors (BM25) for keyword search
- **Efficient Retrieval**: Optimized indexing and querying

### 💾 Conversation Management
- **History Tracking**: Save and load past queries and answers
- **Context Preservation**: Store full conversation with metadata
- **Download Options**: Export results as Markdown or JSON

### 📊 Advanced Features
- **Re-ranking**: Optional cross-encoder re-ranking (bge-reranker-large)
- **Visual Results**: Display associated images with retrieved chunks
- **Metadata Display**: Show detailed metadata for each chunk
- **Performance Metrics**: View search scores and retrieval methods

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     Streamlit UI (app_streamlit.py) │
│  Interactive RAG Query Interface    │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼────────┐  ┌────▼──────────┐
│ Retrieval     │  │ LLM (Ollama)  │
│ Pipeline      │  │ Local Models  │
└──────┬────────┘  └───────────────┘
       │
┌──────▼──────────────────────────┐
│  Vector Database (Qdrant)       │
│  • Dense indices (semantic)     │
│  • Sparse indices (keyword)     │
└──────┬──────────────────────────┘
       │
┌──────▼──────────────────────────┐
│  Data Pipeline                  │
│  • Document Processing          │
│  • Chunking                     │
│  • Metadata Extraction          │
│  • Image Processing             │
└─────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Ollama (for local LLM)
- Qdrant (vector database)
- pip

### Setup Steps

1. **Clone or Download the Repository**
   ```bash
   cd srs-rag-engine
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Required Services**

   **Ollama** (LLM Server):
   ```bash
   # Download from https://ollama.ai
   # Run with: ollama serve
   # Pull models: ollama pull mxbai-embed-large
   ```

   **Qdrant** (Vector Database):
   ```bash
   # Start using Docker Compose
   cd vectordb
   docker-compose up -d qdrant
   ```

## ⚙️ Configuration

### Server URLs

Edit the configuration at the top of `app_streamlit.py`:

```python
OLLAMA_URL = "http://10.117.100.61:11434"      # Ollama server
QDRANT_URL = "http://10.188.105.70:6333"       # Qdrant server
```

### Embedding Models

Configure in `ingestion/hybrid_ingestion.py`:

```python
EMBEDDING_MODEL = "mxbai-embed-large"          # Dense embedding model
SPARSE_MODEL_NAME = "Qdrant/bm25"              # Sparse embedding model
```

### Chunk Directory

When running the ingestion script, specify chunk directories:

```bash
python ingestion/hybrid_ingestion.py --chunk-dir path/to/chunks
# Multiple directories:
python ingestion/hybrid_ingestion.py --chunk-dir dir1 --chunk-dir dir2
```

## 🚀 Usage

### 1. Data Ingestion

**Process Documents and Create Embeddings**

```bash
python ingestion/hybrid_ingestion.py \
  --chunk-dir ./JEA-04135/document_output/chunks \
  --create-collection \
  --vector-size 1024
```

**Options:**
- `--chunk-dir`: Path to chunks directory (can specify multiple times)
- `--create-collection`: Create/recreate Qdrant collection
- `--vector-size`: Embedding vector dimension (default: 1024)
- `--skip-health-check`: Skip server health verification

### 2. Launch Web UI

**Start the Streamlit Application**

```bash
streamlit run app_streamlit.py
```

Access at: `http://localhost:8501`

### 3. Query the System

1. **Configure Search Strategy** (sidebar):
   - Select LLM model
   - Choose search strategy (Dense/Sparse/Hybrid)
   - Set retrieval parameters (top-k, weights)
   - Enable re-ranking if needed

2. **Enter Question**:
   - Type your question about the SRS document

3. **Review Results**:
   - View retrieved chunks with scores
   - See associated images
   - Read LLM-generated answer
   - Download results

## 📁 Project Structure

```
srs-rag-engine/
├── app_streamlit.py              # Main Streamlit UI
├── launcher.py                   # Application launcher
├── process_doc.py                # Document processing utilities
│
├── document_processing/
│   ├── __init__.py
│   ├── chunking_process.py       # Document chunking
│   ├── image_processing.py       # Image extraction & processing
│   ├── tables_processing.py      # Table handling
│   └── vlm_interface.py          # Vision Language Model integration
│
├── ingestion/
│   ├── __init__.py
│   ├── dense_ingestion.py        # Dense embedding ingestion
│   ├── hybrid_ingestion.py       # Hybrid (dense+sparse) ingestion
│
├── retrieval/
│   ├── __init__.py
│   └── retrieval_pipeline.py     # Retrieval pipeline (search strategies)
│
├── vectordb/
│   └── qdrant-compose.yml        # Qdrant Docker setup
│
└── README.md                     # This file
```

## 📚 API Documentation

### Retrieval Pipeline

```python
from retrieval.retrieval_pipeline import SRSRetrievalPipeline, SearchConfig, SearchStrategy

# Initialize pipeline
pipeline = SRSRetrievalPipeline()

# Configure search
config = SearchConfig(
    strategy=SearchStrategy.HYBRID_RRF,  # Search strategy
    top_k=5,                             # Number of results
    dense_weight=0.5,                    # Weight for dense (hybrid only)
    sparse_weight=0.5,                   # Weight for sparse (hybrid only)
    rerank_enabled=False,                # Enable re-ranking
    rerank_model="bge-reranker-large"    # Re-ranker model
)

# Execute search
results = pipeline.search("Your question here", config)

# Results structure
for result in results:
    print(f"Chunk ID: {result.chunk_id}")
    print(f"Score: {result.score}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}")
    print(f"Images: {result.image_groups}")
    print(f"Method: {result.retrieval_method}")
```

### Document Processing

```python
from document_processing.chunking_process import chunk_document
from document_processing.image_processing import extract_images

# Chunk document
chunks = chunk_document(doc_path, chunk_size=512, overlap=50)

# Extract images
images = extract_images(doc_path)
```

## 🔧 Troubleshooting

### Image Files Not Found

**Error:** `⚠️ File not found: JEA-04135/document_output/filtered_images/image_2.png`

**Solutions:**
1. Verify image exists at path: `<workspace_root>/JEA-04135/document_output/filtered_images/image_2.png`
2. Move `app_streamlit.py` to workspace root if using relative paths
3. Check that paths stored in Qdrant match actual file locations

### Cannot Connect to Ollama

**Error:** `❌ Cannot connect to Ollama at http://...`

**Solutions:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Verify correct URL in app_streamlit.py
# Restart Ollama if needed
ollama serve
```

### Cannot Connect to Qdrant

**Error:** `❌ Cannot reach Qdrant server`

**Solutions:**
```bash
# Start Qdrant with Docker Compose
cd vectordb
docker-compose up -d qdrant

# Verify connection
curl http://localhost:6333/health
```

### No Chunks Found During Ingestion

**Solutions:**
1. Verify chunk directory exists and contains `chunk_*.txt` files
2. Use absolute paths for reliability
3. Check file permissions

```bash
# List available chunks
ls -la JEA-04135/document_output/chunks/chunk_*.txt
```

### Memory/Performance Issues

**Optimize for large datasets:**
- Reduce `BATCH_SIZE` in ingestion script
- Lower `top_k` in search configuration
- Use sparse-only search instead of hybrid

## 📊 System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum (16GB+ recommended)
- **Disk**: 50GB+ (depends on document size)
- **Network**: For local deployment only

## 🔐 Security Notes

- ✅ All processing is local (no cloud uploads)
- ✅ No external API calls for LLM
- ✅ Queries stored only in browser session
- ✅ Conversation history stored locally

## 📝 License

[Add your license information here]

## 👥 Support

For issues and questions:
1. Check the Troubleshooting section
2. Review logs in the terminal
3. Verify server connectivity
4. Check configuration settings

## 🎯 Roadmap

- [ ] Multi-document support
- [ ] Advanced query expansion
- [ ] Fine-tuned re-ranking models
- [ ] Semantic caching
- [ ] API endpoint exposure
- [ ] Batch query processing
- [ ] Export to different formats

---

**Last Updated:** June 2026  
**Version:** 1.0.0
