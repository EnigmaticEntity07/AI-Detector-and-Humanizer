import os
import random
import logging
import chromadb
from chromadb.config import Settings
from data_loader import load_and_harmonize_datasets

logger = logging.getLogger(__name__)

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "human_texts"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def build_vector_index(data_dir: str = DATA_DIR, max_texts: int = 2500):
    """
    Builds the vector index from human-written texts in the dataset.
    """
    client = get_chroma_client()
    
    # Attempt to clear existing collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
        
    collection = client.create_collection(name=COLLECTION_NAME)
    
    logger.info(f"Loading data from {data_dir} to build vector index...")
    df = load_and_harmonize_datasets(data_dir)
    if df.empty:
        logger.warning("No data found to build index.")
        return
        
    human_texts = df[df["label"] == 0]["text"].dropna().tolist()
    
    if len(human_texts) > max_texts:
        logger.info(f"Sampling {max_texts} texts from {len(human_texts)} total human records.")
        human_texts = random.sample(human_texts, max_texts)
        
    logger.info(f"Indexing {len(human_texts)} human texts into ChromaDB...")
    
    batch_size = 500
    for i in range(0, len(human_texts), batch_size):
        batch = human_texts[i:i+batch_size]
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        collection.add(
            documents=batch,
            ids=ids
        )
    logger.info("Indexing complete.")

def query_human_examples(query_text: str, n: int = 3, max_chars: int = 500) -> list[str]:
    """
    Retrieves semantically similar human examples.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        logger.warning("ChromaDB collection not found. Make sure to build the index first.")
        return []

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
            
        docs = results["documents"][0]
        
        # Truncate to avoid blowing up context window
        truncated = []
        for s in docs:
            if len(s) > max_chars:
                cut = s[:max_chars]
                last_period = cut.rfind(".")
                if last_period > max_chars // 2:
                    cut = cut[: last_period + 1]
                truncated.append(cut)
            else:
                truncated.append(s)
                
        return truncated
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_vector_index()
