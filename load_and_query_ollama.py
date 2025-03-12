import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.llms import Ollama

# Configuration
EMBEDDINGS_DIR = 'generated_embeddings'
COLLECTION_NAME_CODE = 'code_collection'
COLLECTION_NAME_DOC = 'doc_collection'
BATCH_SIZE = 100  # For Qdrant insertion
OLLAMA_MODEL = "deepseek-r1:1.5b"  # Local Ollama model

def load_saved_data():
    """Load previously generated embeddings and snippets."""
    print(f"Loading data from {EMBEDDINGS_DIR}...")
    
    # Load snippets
    with open(os.path.join(EMBEDDINGS_DIR, 'snippets.pkl'), 'rb') as f:
        snippets_by_type = pickle.load(f)
    
    # Load code embeddings
    code_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, 'code_embeddings.npy'))
    
    # Load doc embeddings
    doc_embeddings = np.load(os.path.join(EMBEDDINGS_DIR, 'doc_embeddings.npy'))
    
    # Load metadata
    with open(os.path.join(EMBEDDINGS_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    code_snippets = snippets_by_type['code']
    doc_snippets = snippets_by_type['doc']
    
    print(f"Loaded {len(code_snippets)} code snippets and {len(doc_snippets)} documentation snippets.")
    print(f"Code embedding dimension: {metadata['code_dim']}")
    print(f"Doc embedding dimension: {metadata['doc_dim']}")
    
    return snippets_by_type, {'code': code_embeddings, 'doc': doc_embeddings}, metadata

def setup_qdrant(metadata):
    """Set up Qdrant client and collections for code and documentation."""
    print("Setting up Qdrant...")
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    
    # Setup code collection
    if client.collection_exists(COLLECTION_NAME_CODE):
        client.delete_collection(COLLECTION_NAME_CODE)
        print(f"Deleted existing collection '{COLLECTION_NAME_CODE}'.")
    
    client.create_collection(
        collection_name=COLLECTION_NAME_CODE,
        vectors_config=VectorParams(
            size=metadata['code_dim'],
            distance=Distance.COSINE,
        )
    )
    print(f"Created new collection '{COLLECTION_NAME_CODE}'.")
    
    # Setup documentation collection
    if client.collection_exists(COLLECTION_NAME_DOC):
        client.delete_collection(COLLECTION_NAME_DOC)
        print(f"Deleted existing collection '{COLLECTION_NAME_DOC}'.")
    
    client.create_collection(
        collection_name=COLLECTION_NAME_DOC,
        vectors_config=VectorParams(
            size=metadata['doc_dim'],
            distance=Distance.COSINE,
        )
    )
    print(f"Created new collection '{COLLECTION_NAME_DOC}'.")
    
    return client

def insert_embeddings(client, snippets_by_type, embeddings_by_type, batch_size=100):
    """Insert embeddings into Qdrant in batches."""
    print("Inserting code embeddings into Qdrant...")
    
    # Insert code embeddings
    code_snippets = snippets_by_type['code']
    code_embeddings = embeddings_by_type['code']
    
    for i in range(0, len(code_snippets), batch_size):
        batch_snippets = code_snippets[i:i+batch_size]
        batch_embeddings = code_embeddings[i:i+batch_size]
        
        points = [
            PointStruct(
                id=i + idx,
                vector=embedding.tolist(),
                payload={"file": snippet["file"], "content": snippet["content"], "type": "code"}
            )
            for idx, (snippet, embedding) in enumerate(zip(batch_snippets, batch_embeddings))
        ]
        
        client.upsert(collection_name=COLLECTION_NAME_CODE, points=points)
        print(f"Inserted code batch {i//batch_size + 1} ({len(points)} points).")
    
    print("All code embeddings inserted successfully.")
    
    # Insert doc embeddings
    print("Inserting documentation embeddings into Qdrant...")
    doc_snippets = snippets_by_type['doc']
    doc_embeddings = embeddings_by_type['doc']
    
    for i in range(0, len(doc_snippets), batch_size):
        batch_snippets = doc_snippets[i:i+batch_size]
        batch_embeddings = doc_embeddings[i:i+batch_size]
        
        points = [
            PointStruct(
                id=i + idx,
                vector=embedding.tolist(),
                payload={"file": snippet["file"], "content": snippet["content"], "type": "doc"}
            )
            for idx, (snippet, embedding) in enumerate(zip(batch_snippets, batch_embeddings))
        ]
        
        client.upsert(collection_name=COLLECTION_NAME_DOC, points=points)
        print(f"Inserted doc batch {i//batch_size + 1} ({len(points)} points).")
    
    print("All documentation embeddings inserted successfully.")

def perform_query_with_langchain(client, query_text, metadata, code_limit=15, doc_limit=5):
    """Perform a semantic search query across both code and documentation collections with Ollama via LangChain."""
    print(f"Querying with LangChain: '{query_text}'")
    
    # Initialize the appropriate embedding models
    code_encoder = SentenceTransformer(metadata['code_model'])
    doc_encoder = SentenceTransformer(metadata['doc_model'])
    
    # Generate embeddings for the query using both models
    code_query_embedding = code_encoder.encode(query_text, normalize_embeddings=True)
    doc_query_embedding = doc_encoder.encode(query_text, normalize_embeddings=True)
    
    # Perform queries on both collections
    code_results = client.query_points(
        collection_name=COLLECTION_NAME_CODE,
        query=code_query_embedding.tolist(),
        limit=code_limit,
        with_payload=True
    )
    
    doc_results = client.query_points(
        collection_name=COLLECTION_NAME_DOC,
        query=doc_query_embedding.tolist(),
        limit=doc_limit,
        with_payload=True
    )
    
    # Prepare context from retrieved documents
    context_parts = []
    
    # Add code results with full file context
    print(f"\nRetrieved Code Files ({len(code_results.points)}):")
    for i, point in enumerate(code_results.points):
        file_path = point.payload["file"]
        content = point.payload["content"]
        
        # Format the context entry
        context_entry = f"[CODE - {file_path}]\n{content}\n"
        context_parts.append(context_entry)
        print(f"{i+1}. {file_path} (similarity: {point.score:.4f})")
    
    # Add documentation results with full file context
    print(f"\nRetrieved Documentation Files ({len(doc_results.points)}):")
    for i, point in enumerate(doc_results.points):
        file_path = point.payload["file"]
        content = point.payload["content"]
        
        # Format the context entry
        context_entry = f"[DOC - {file_path}]\n{content}\n"
        context_parts.append(context_entry)
        print(f"{i+1}. {file_path} (similarity: {point.score:.4f})")
    
    context = "\n".join(context_parts)
    
    # Initialize Ollama LLM via LangChain
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.6,
        top_p=0.7
    )

    # Generate response using Ollama
    prompt_template = f"""
    You are an assistant tasked with answering questions about a codebase. Use the provided context to answer the question concisely.
    
    The context includes both source code and documentation files. Pay attention to the type of each file (CODE or DOC).
    
    Context:
    {context}

    Question:
    {query_text}

    Answer:
    """
    
    print("Sending prompt to Ollama...")
    response = llm.invoke(prompt_template)
    
    print("\nGenerated Answer:")
    print(response)
    
    return response

def perform_query_with_direct_api(client, query_text, metadata, code_limit=15, doc_limit=5):
    """Perform a semantic search query across both code and documentation collections with Ollama via direct API."""
    import requests
    
    print(f"Querying with direct API: '{query_text}'")
    
    # Initialize the appropriate embedding models
    code_encoder = SentenceTransformer(metadata['code_model'])
    doc_encoder = SentenceTransformer(metadata['doc_model'])
    
    # Generate embeddings for the query using both models
    code_query_embedding = code_encoder.encode(query_text, normalize_embeddings=True)
    doc_query_embedding = doc_encoder.encode(query_text, normalize_embeddings=True)
    
    # Perform queries on both collections
    code_results = client.query_points(
        collection_name=COLLECTION_NAME_CODE,
        query=code_query_embedding.tolist(),
        limit=code_limit,
        with_payload=True
    )
    
    doc_results = client.query_points(
        collection_name=COLLECTION_NAME_DOC,
        query=doc_query_embedding.tolist(),
        limit=doc_limit,
        with_payload=True
    )
    
    # Prepare context from retrieved documents
    context_parts = []
    
    # Add code results with full file context
    print(f"\nRetrieved Code Files ({len(code_results.points)}):")
    for i, point in enumerate(code_results.points):
        file_path = point.payload["file"]
        content = point.payload["content"]
        
        # Format the context entry
        context_entry = f"[CODE - {file_path}]\n{content}\n"
        context_parts.append(context_entry)
        print(f"{i+1}. {file_path} (similarity: {point.score:.4f})")
    
    # Add documentation results with full file context
    print(f"\nRetrieved Documentation Files ({len(doc_results.points)}):")
    for i, point in enumerate(doc_results.points):
        file_path = point.payload["file"]
        content = point.payload["content"]
        
        # Format the context entry
        context_entry = f"[DOC - {file_path}]\n{content}\n"
        context_parts.append(context_entry)
        print(f"{i+1}. {file_path} (similarity: {point.score:.4f})")
    
    context = "\n".join(context_parts)
    
    # Create prompt for Ollama
    prompt_template = f"""
    You are an assistant tasked with answering questions about a codebase. Use the provided context to answer the question concisely.
    
    The context includes both source code and documentation files. Pay attention to the type of each file (CODE or DOC).
    
    Context:
    {context}

    Question:
    {query_text}

    Answer:
    """
    
    # Call Ollama API directly
    print("Sending request to Ollama API...")
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': OLLAMA_MODEL,
            'prompt': prompt_template,
            'temperature': 0.6,
            'top_p': 0.7
        }
    )
    
    result = response.json()['response']
    
    print("\nGenerated Answer:")
    print(result)
    
    return result

def main():
    # Load saved data
    snippets_by_type, embeddings_by_type, metadata = load_saved_data()
    
    # Set up Qdrant
    client = setup_qdrant(metadata)
    
    # Insert embeddings into Qdrant
    insert_embeddings(client, snippets_by_type, embeddings_by_type, BATCH_SIZE)
    
    # Ask user which interface they prefer
    print("\nChoose an interface to Ollama:")
    print("1. LangChain (requires langchain-community package)")
    print("2. Direct API (uses requests package)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    # Interactive query loop with Ollama integration
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        if choice == "1":
            perform_query_with_langchain(client, query, metadata)
        else:
            perform_query_with_direct_api(client, query, metadata)

if __name__ == "__main__":
    main()
