# RAGBase

_A advanced Retrieval Augmented Generation framework designed to enhance your AI applications with multi-source knowledge retrieval._

## Overview

RAGBase provides a modular, extensible solution for building RAG systems with multiple retrieval agents targeting different knowledge sources. By leveraging vector databases and optimized embeddings, RAGBase delivers highly relevant context to your LLM, resulting in more accurate, knowledge-grounded responses.

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

RAGBase implements three specialized retrieval agents:

- **Source Code Retriever**  
  Intelligently navigates codebases to extract relevant functions, classes, and implementation details. Optimized for understanding code semantics beyond simple keyword matching.
- **README Doc Retriever**  
  Specializes in extracting high-level project information, setup instructions, and usage examples from documentation. Particularly valuable for understanding project architecture and intent.
- **Conversation Document Retriever**  
  Maintains and indexes past interactions, allowing the system to reference previous questions and answers for improved context awareness and continuity.

Each agent can operate independently or together, with the system dynamically determining which knowledge sources are most relevant for each query.

