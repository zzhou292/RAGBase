import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For progress bars

# Configuration
FOLDER_PATH = 'chrono'
ALLOWED_EXTENSIONS = ['.cpp', '.hpp', '.py', '.md']
OUTPUT_DIR = 'generated_embeddings'
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # For embedding generation

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_codebase(folder_path, allowed_extensions):
    """Load code snippets from the codebase."""
    snippets = []
    print("Loading codebase...")
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in allowed_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")
                        continue
                
                # Only add non-empty files
                if content.strip():
                    snippets.append({"file": file_path, "content": content})
    
    return snippets

def generate_embeddings(snippets, model_name, batch_size=32):
    """Generate embeddings for code snippets."""
    print(f"Generating embeddings using {model_name}...")
    
    # Initialize the embedding model
    encoder = SentenceTransformer(model_name)
    
    # Extract text content for embedding
    texts = [snippet["content"] for snippet in snippets]
    
    # Generate embeddings in batches with progress bar
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = encoder.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings), encoder.get_sentence_embedding_dimension()

def save_data(snippets, embeddings, embedding_dim):
    """Save snippets, embeddings, and metadata to disk."""
    print(f"Saving data to {OUTPUT_DIR}...")
    
    # Save code snippets
    with open(os.path.join(OUTPUT_DIR, 'code_snippets.pkl'), 'wb') as f:
        pickle.dump(snippets, f)
    
    # Save embeddings as numpy array
    np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
    
    # Save metadata (embedding dimension, model name, etc.)
    metadata = {
        'embedding_dim': embedding_dim,
        'model_name': MODEL_NAME,
        'num_snippets': len(snippets),
        'allowed_extensions': ALLOWED_EXTENSIONS
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Successfully saved {len(snippets)} code snippets and their embeddings.")
    print(f"Embedding dimension: {embedding_dim}")

def main():
    # Load code snippets
    code_snippets = load_codebase(FOLDER_PATH, ALLOWED_EXTENSIONS)
    print(f"Loaded {len(code_snippets)} code snippets.")
    
    # Generate embeddings
    embeddings, embedding_dim = generate_embeddings(code_snippets, MODEL_NAME, BATCH_SIZE)
    print(f"Generated embeddings of shape: {embeddings.shape}")
    
    # Save data to disk
    save_data(code_snippets, embeddings, embedding_dim)

if __name__ == "__main__":
    main()
