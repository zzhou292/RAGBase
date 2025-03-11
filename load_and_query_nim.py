import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Configuration
EMBEDDINGS_DIR = 'generated_embeddings'
COLLECTION_NAME = 'codebase_collection'
BATCH_SIZE = 100  # For Qdrant insertion
MODEL_NAME = "meta/llama3-70b-instruct"  # NVIDIA LLM model

def load_saved_data():
    """Load previously generated embeddings and code snippets."""
    print(f"Loading data from {EMBEDDINGS_DIR}...")
    
    # Load code snippets
    with open(os.path.join(EMBEDDINGS_DIR, 'code_snippets.pkl'), 'rb') as f:
        snippets = pickle.load(f)
    
    # Load embeddings
    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, 'embeddings.npy'))
    
    # Load metadata
    with open(os.path.join(EMBEDDINGS_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {len(snippets)} code snippets and their embeddings.")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    
    return snippets, embeddings, metadata

def setup_qdrant(embedding_dim):
    """Set up Qdrant client and collection."""
    print("Setting up Qdrant...")
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    
    # Check if collection exists and recreate it
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    
    # Create new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        )
    )
    print(f"Created new collection '{COLLECTION_NAME}'.")
    
    return client

def insert_embeddings(client, snippets, embeddings, batch_size=100):
    """Insert embeddings into Qdrant in batches."""
    print("Inserting embeddings into Qdrant...")
    
    for i in range(0, len(snippets), batch_size):
        batch_snippets = snippets[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        points = [
            PointStruct(
                id=i + idx,
                vector=embedding.tolist(),
                payload={"file": snippet["file"], "content": snippet["content"]}
            )
            for idx, (snippet, embedding) in enumerate(zip(batch_snippets, batch_embeddings))
        ]
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Inserted batch {i//batch_size + 1} ({len(points)} points).")
    
    print("All embeddings inserted successfully.")

def perform_query(client, query_text, model_name, limit=25):
    """Perform a semantic search query and use ChatNVIDIA to generate an answer."""
    print(f"Querying: '{query_text}'")
    
    # Initialize the embedding model for the query
    encoder = SentenceTransformer(model_name)
    
    # Generate embedding for the query
    query_embedding = encoder.encode(query_text)
    
    # Perform query on Qdrant
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    )
    
    # Prepare context from retrieved documents
    context = "\n".join([point.payload["content"][:1000] for point in search_results.points])
    
    print("\nRetrieved Context:")
    print(context)
    
    # Initialize ChatNVIDIA LLM
    llm = ChatNVIDIA(
        model="deepseek-ai/deepseek-r1-distill-qwen-32b",
        api_key="PASTE YOUR API KEY HERE", 
        temperature=0.6,
        top_p=0.7,
        max_tokens=18192,
        )

    # Generate response using ChatNVIDIA
    prompt_template = f"""
        You are an assistant tasked with answering questions based on the provided context. Use the context below to answer the question concisely.

        Context:
        {context}

        Question:
        {query_text}

        Answer:
        """
    
    print("Prompt:")
    print(prompt_template)
    
    response = llm.invoke(prompt_template)
    
    print("\nGenerated Answer:")
    print(response)

def main():
    # Load saved data
    snippets, embeddings, metadata = load_saved_data()
    
    # Set up Qdrant
    client = setup_qdrant(metadata['embedding_dim'])
    
    # Insert embeddings into Qdrant
    insert_embeddings(client, snippets, embeddings, BATCH_SIZE)
    
    # Interactive query loop with ChatNVIDIA integration
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        perform_query(client, query, metadata['model_name'], limit=5)

if __name__ == "__main__":
    main()
