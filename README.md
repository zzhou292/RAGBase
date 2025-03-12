# RAGBase4Code

_A advanced Retrieval Augmented Generation framework designed to enhance your AI applications with multi-source knowledge retrieval. Specifically designed for offline LLM4CODE, aiming to provide codebase understanding offline and securely._

# Author

This codebase is designed and maintained by Json Zhou (zzhou292@wisc.edu).

## Overview

RAGBase4Code provides a modular, extensible solution for building RAG systems with multiple retrieval agents targeting different knowledge sources. By leveraging vector databases and optimized embeddings, RAGBase delivers highly relevant context to your LLM, resulting in more accurate, knowledge-grounded responses.

Yes, you can understand this as a backbone of a secured Cursor AI (https://www.cursor.com/) or Trae (https://www.trae.ai/).

### System Phases

- **Data Preparation & Indexing**  
  Process various data sources into searchable vector embeddings.
- **Retrieval & Generation**  
  Dynamically fetch relevant information and generate responses when users ask questions.

## Key Features

- **Triple-Agent Retrieval System**  
  Specialized agents for source code, documentation, and conversation history.
- **Multiple LLM Interfaces**  
  Seamless integration with NVIDIA NIM services and Ollama for local model deployment.
- **Fully Customizable Pipelines**  
  Configure every aspect of the RAG workflow from chunking to prompt engineering.
- **Scalable Architecture**  
  Built to handle everything from personal projects to enterprise-scale applications.
- **Privacy-Focused**  
  Process sensitive data locally without external API calls when using Ollama integration.
- **Comprehensive Evaluation**  
  Built-in metrics for measuring retrieval quality and response accuracy.

## Architecture

RAGBase follows a modular architecture with separate components for:

- **Document Processing**  
  Extract, clean, and chunk text from various sources.
- **Vector Embedding**  
  Convert text into numerical representations using configurable embedding models.
- **Vector Storage**  
  Index and store embeddings for efficient similarity search.
- **Retrieval Agents**  
  Specialized components to extract relevant information based on query context.
- **LLM Interface**  
  Connect to either NVIDIA NIM services or local Ollama models.
- **Response Generator**  
  Combine retrieved information with the query to produce accurate answers.

## Retrieval Agents

RAGBase4Code implements three specialized retrieval agents:

- **Source Code Retriever**  
  Intelligently navigates codebases to extract relevant functions, classes, and implementation details. Optimized for understanding code semantics beyond simple keyword matching.
- **README Doc Retriever**  
  Specializes in extracting high-level project information, setup instructions, and usage examples from documentation. Particularly valuable for understanding project architecture and intent.
- **Conversation Document Retriever**  
  Maintains and indexes past interactions, allowing the system to reference previous questions and answers for improved context awareness and continuity.

Each agent can operate independently or together, with the system dynamically determining which knowledge sources are most relevant for each query.

## Instructions

### Prerequisites & Installations

- **Ollama (for local LLM deployment):**  
  • If you have not already installed Ollama, download the appropriate installer from the [official Ollama website](https://ollama.com).  
  • After installing, you can start the background service with:
  ```
  'ollama serve'
  ```
  (You may adjust or disable autostart settings based on your preferences.)

- **Sentence Transformers (for generating embeddings):**  
Install the Sentence Transformers library using pip:
```
pip3 install -U sentence-transformers
```
- **Qdrant (via Docker, for vector storage and similarity search):**  
Make sure Docker is installed and running on your system. Then launch Qdrant in the background with:

```
docker run -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

This command ensures that Qdrant runs with persistent storage mapped to the `qdrant_storage` directory in your current path.

### Running the Pipeline

1. **Generate Embeddings:**  
 Run the script to process your data and generate vector embeddings:

  ```
python generate_embeddings.py
  ```

2. **Query the System:**  
After generating embeddings, choose the appropriate script based on your LLM interface:  
- For NVIDIA NIM services:
  ```
  python load_and_query_nim.py
  ```
- For local Ollama deployment:
  ```
  python load_and_query_ollama.py
  ```

Follow these steps sequentially to set up your environment and execute the full RAG pipeline.

  

