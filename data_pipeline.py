import os
import glob
import json
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "unified_dataset.csv")
BLACKLIST_JSON = os.path.join(DATA_DIR, "dynamic_blacklist.json")

def load_and_clean_datasets():
    all_dfs = []
    
    # 1. AIvsHuman1
    try:
        df1 = pd.read_csv(os.path.join(BASE_DIR, "AIvsHuman1", "data_for_preprocessing.csv"))
        df1 = df1[['Text', 'Author']].rename(columns={'Text': 'text', 'Author': 'label'})
        df1['label'] = df1['label'].apply(lambda x: 1 if str(x).strip().lower() == 'ai' else 0)
        all_dfs.append(df1)
    except Exception as e:
        logger.error(f"Error loading AIvsHuman1: {e}")

    # 2. AIvsHuman2
    try:
        df2 = pd.read_csv(os.path.join(BASE_DIR, "AIvsHuman2", "ai_vs_human_text_2026.csv"))
        df2 = df2[['text_content', 'label']].rename(columns={'text_content': 'text'})
        df2['label'] = df2['label'].apply(lambda x: 1 if str(x).strip().lower() == 'ai' else (0 if str(x).strip().lower() == 'human' else pd.to_numeric(x, errors='coerce')))
        all_dfs.append(df2)
    except Exception as e:
        logger.error(f"Error loading AIvsHuman2: {e}")

    # 3. AIvsHuman3
    try:
        df3 = pd.read_csv(os.path.join(BASE_DIR, "AIvsHuman3", "balanced_ai_human_prompts.csv"))
        df3 = df3[['text', 'generated']].rename(columns={'generated': 'label'})
        all_dfs.append(df3)
    except Exception as e:
        logger.error(f"Error loading AIvsHuman3: {e}")

    # 4. AIvsHuman4
    try:
        df4 = pd.read_csv(os.path.join(BASE_DIR, "AIvsHuman4", "ai_human_content_detection_dataset.csv"))
        df4 = df4[['text_content', 'label']].rename(columns={'text_content': 'text'})
        all_dfs.append(df4)
    except Exception as e:
        logger.error(f"Error loading AIvsHuman4: {e}")

    # 5. AIvsHuman5
    try:
        df5 = pd.read_csv(os.path.join(BASE_DIR, "AIvsHuman5", "train_v2_drcat_02.csv"))
        df5 = df5[['text', 'label']]
        all_dfs.append(df5)
    except Exception as e:
        logger.error(f"Error loading AIvsHuman5: {e}")

    if not all_dfs:
        logger.error("No datasets could be loaded.")
        return None

    # Merge and clean
    unified = pd.concat(all_dfs, ignore_index=True)
    unified = unified.dropna(subset=['text', 'label'])
    unified['label'] = pd.to_numeric(unified['label'], errors='coerce')
    unified = unified.dropna(subset=['label'])
    unified['label'] = unified['label'].astype(int)
    
    # Filter only 0 and 1
    unified = unified[unified['label'].isin([0, 1])]
    
    # Drop duplicates
    unified = unified.drop_duplicates(subset=['text'])
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    unified.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved unified dataset with {len(unified)} records to {OUTPUT_CSV}")
    return unified

def generate_dynamic_blacklist(df, top_n=50):
    logger.info("Generating dynamic TF-IDF blacklist...")
    
    # Sample down if too large to save memory during TF-IDF
    if len(df) > 20000:
        df_sample = df.groupby('label').apply(lambda x: x.sample(10000, random_state=42)).reset_index(drop=True)
    else:
        df_sample = df
        
    # Extract AI and Human texts
    ai_texts = df_sample[df_sample['label'] == 1]['text'].astype(str).tolist()
    human_texts = df_sample[df_sample['label'] == 0]['text'].astype(str).tolist()
    
    if not ai_texts or not human_texts:
        logger.error("Need both AI and Human texts to generate blacklist.")
        return
        
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    
    # Fit on all texts
    all_texts = ai_texts + human_texts
    vectorizer.fit(all_texts)
    
    # Transform separately
    ai_tfidf = vectorizer.transform(ai_texts)
    human_tfidf = vectorizer.transform(human_texts)
    
    # Get mean TF-IDF scores for each word
    ai_mean = np.asarray(ai_tfidf.mean(axis=0)).flatten()
    human_mean = np.asarray(human_tfidf.mean(axis=0)).flatten()
    
    # Find words where AI mean > Human mean
    diff = ai_mean - human_mean
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top N indices with highest difference in favor of AI
    top_indices = diff.argsort()[-top_n:][::-1]
    
    top_words = [feature_names[i] for i in top_indices if diff[i] > 0]
    
    # Save to JSON
    with open(BLACKLIST_JSON, "w") as f:
        json.dump(top_words, f, indent=4)
        
    logger.info(f"Saved {len(top_words)} dynamic blacklist words to {BLACKLIST_JSON}")
    logger.info(f"Top 10 AI words: {top_words[:10]}")

if __name__ == "__main__":
    df = load_and_clean_datasets()
    if df is not None:
        generate_dynamic_blacklist(df)
