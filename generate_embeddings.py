import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For progress bars

# Configuration
FOLDER_PATH = 'example_codebase/chrono'
CODE_EXTENSIONS = ['.cpp', '.hpp', '.py']
DOC_EXTENSIONS = ['.md']
ALLOWED_EXTENSIONS = CODE_EXTENSIONS + DOC_EXTENSIONS
OUTPUT_DIR = 'generated_embeddings'
CODE_MODEL = "microsoft/codebert-base"  # For code files
DOC_MODEL = "sentence-transformers/all-mpnet-base-v2"  # For documentation files
BATCH_SIZE = 32  # For embedding generation

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_codebase(folder_path, allowed_extensions):
    """Load code snippets from the codebase with file type annotations."""
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
                    # Determine file type based on extension
                    file_ext = os.path.splitext(file)[1]
                    file_type = "code" if file_ext in CODE_EXTENSIONS else "doc"
                    snippets.append({
                        "file": file_path, 
                        "content": content, 
                        "type": file_type
                    })
    
    return snippets

def generate_code_embeddings(code_snippets):
    """Generate embeddings for code snippets using SFR-Embedding-Code model."""
    print(f"Generating code embeddings using {CODE_MODEL}...")
    
    # Initialize the SFR-Embedding-Code model
    encoder = SentenceTransformer(CODE_MODEL, trust_remote_code=True)
    
    # Extract text content for embedding
    texts = [snippet["content"] for snippet in code_snippets]
    
    # Generate embeddings in batches with progress bar
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        # For SFR-Embedding-Code, we don't need a special prompt for code
        batch_embeddings = encoder.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings), encoder.get_sentence_embedding_dimension()

def generate_doc_embeddings(doc_snippets):
    """Generate embeddings for documentation using selected model."""
    print(f"Generating documentation embeddings using {DOC_MODEL}...")
    
    # Initialize the model for documentation
    encoder = SentenceTransformer(DOC_MODEL)
    
    # Extract text content for embedding
    texts = [snippet["content"] for snippet in doc_snippets]
    
    # Generate embeddings in batches with progress bar
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_embeddings = encoder.encode(
            batch_texts,
            normalize_embeddings=True,  # For better similarity calculations
            show_progress_bar=False
        )
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings), encoder.get_sentence_embedding_dimension()

def save_data(snippets_by_type, embeddings_by_type, metadata):
    """Save snippets, embeddings, and metadata to disk."""
    print(f"Saving data to {OUTPUT_DIR}...")
    
    # Save all snippets
    with open(os.path.join(OUTPUT_DIR, 'snippets.pkl'), 'wb') as f:
        pickle.dump({
            'code': snippets_by_type['code'],
            'doc': snippets_by_type['doc']
        }, f)
    
    # Save code embeddings
    np.save(os.path.join(OUTPUT_DIR, 'code_embeddings.npy'), embeddings_by_type['code'])
    
    # Save doc embeddings
    np.save(os.path.join(OUTPUT_DIR, 'doc_embeddings.npy'), embeddings_by_type['doc'])
    
    # Save metadata
    with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    total_snippets = len(snippets_by_type['code']) + len(snippets_by_type['doc'])
    print(f"Successfully saved {total_snippets} snippets and their embeddings.")
    print(f"Code embedding dimension: {metadata['code_dim']}")
    print(f"Doc embedding dimension: {metadata['doc_dim']}")

def main():
    # Load code snippets
    all_snippets = load_codebase(FOLDER_PATH, ALLOWED_EXTENSIONS)
    print(f"Loaded {len(all_snippets)} snippets.")
    
    # Separate code and documentation snippets
    code_snippets = [snippet for snippet in all_snippets if snippet["type"] == "code"]
    doc_snippets = [snippet for snippet in all_snippets if snippet["type"] == "doc"]
    
    print(f"Found {len(code_snippets)} code files and {len(doc_snippets)} documentation files.")
    
    # Generate embeddings for each type
    code_embeddings, code_dim = generate_code_embeddings(code_snippets)
    doc_embeddings, doc_dim = generate_doc_embeddings(doc_snippets)
    
    print(f"Generated code embeddings shape: {code_embeddings.shape}")
    print(f"Generated doc embeddings shape: {doc_embeddings.shape}")
    
    # Organize results by type
    snippets_by_type = {
        'code': code_snippets,
        'doc': doc_snippets
    }
    
    embeddings_by_type = {
        'code': code_embeddings,
        'doc': doc_embeddings
    }
    
    # Create metadata
    metadata = {
        'code_model': CODE_MODEL,
        'doc_model': DOC_MODEL,
        'code_dim': code_dim,
        'doc_dim': doc_dim,
        'code_extensions': CODE_EXTENSIONS,
        'doc_extensions': DOC_EXTENSIONS
    }
    
    # Save data to disk
    save_data(snippets_by_type, embeddings_by_type, metadata)

if __name__ == "__main__":
    main()
