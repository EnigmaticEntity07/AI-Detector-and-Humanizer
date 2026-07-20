import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    dfs = []
    
    # 1. Original Dataset
    if os.path.exists("dataset.csv"):
        logger.info("Reading dataset.csv...")
        df1 = pd.read_csv("dataset.csv")
        if "text" in df1.columns and "label" in df1.columns:
            df1_clean = df1[["text", "label"]].copy()
            dfs.append(df1_clean)
            logger.info(f"Added {len(df1_clean)} rows from dataset.csv")
            
    # 2. AIvsHuman 2026 Dataset
    path2 = "AIvsHuman/ai_vs_human_text_2026.csv"
    if os.path.exists(path2):
        logger.info(f"Reading {path2}...")
        df2 = pd.read_csv(path2)
        if "text_content" in df2.columns and "label" in df2.columns:
            df2_clean = pd.DataFrame()
            df2_clean["text"] = df2["text_content"]
            # 'human' -> 0, anything else -> 1
            df2_clean["label"] = df2["label"].apply(lambda x: 0 if str(x).lower() == 'human' else 1)
            dfs.append(df2_clean)
            logger.info(f"Added {len(df2_clean)} rows from {path2}")
            
    # 3. AIvsHuman2 Dataset
    path3 = "AIvsHuman2/data_for_preprocessing.csv"
    if os.path.exists(path3):
        logger.info(f"Reading {path3}...")
        df3 = pd.read_csv(path3)
        if "Text" in df3.columns and "Author" in df3.columns:
            df3_clean = pd.DataFrame()
            df3_clean["text"] = df3["Text"]
            # 'human' -> 0, anything else -> 1
            df3_clean["label"] = df3["Author"].apply(lambda x: 0 if str(x).lower() == 'human' else 1)
            dfs.append(df3_clean)
            logger.info(f"Added {len(df3_clean)} rows from {path3}")

    if not dfs:
        logger.error("No datasets found or valid!")
        return
        
    logger.info("Concatenating datasets...")
    combined = pd.concat(dfs, ignore_index=True)
    
    logger.info("Dropping duplicates and NaNs...")
    combined = combined.dropna(subset=["text", "label"])
    combined = combined.drop_duplicates(subset=["text"])
    
    logger.info(f"Final combined dataset size: {len(combined)}")
    logger.info(f"Label distribution:\n{combined['label'].value_counts()}")
    
    out_path = "combined_dataset.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}!")

if __name__ == "__main__":
    main()
