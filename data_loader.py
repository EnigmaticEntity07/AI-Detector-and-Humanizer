import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_harmonize_datasets(data_dir: str = "./data") -> pd.DataFrame:
    """
    Scans data_dir for .csv, .csv.gz, and .json files.
    Harmonizes them into a single DataFrame with columns: ['text', 'label', 'dataset_source'].
    Returns the combined DataFrame.
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory '{data_dir}' does not exist.")
        return pd.DataFrame(columns=["text", "label", "dataset_source"])

    all_dfs = []
    
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue

        if file_name.endswith('.csv') or file_name.endswith('.csv.gz'):
            try:
                df = pd.read_csv(file_path)
                if 'text' in df.columns and 'label' in df.columns:
                    df = df[['text', 'label']].copy()
                    df['dataset_source'] = file_name
                    all_dfs.append(df)
                else:
                    logger.warning(f"File {file_name} is missing 'text' or 'label' column.")
            except Exception as e:
                logger.error(f"Failed to read {file_name}: {e}")
                
        elif file_name.endswith('.json'):
            try:
                # Try reading as JSON records first (lines=True)
                try:
                    df = pd.read_json(file_path, lines=True)
                except ValueError:
                    # Fallback to standard JSON array
                    df = pd.read_json(file_path)
                    
                if 'text' in df.columns and 'label' in df.columns:
                    df = df[['text', 'label']].copy()
                    df['dataset_source'] = file_name
                    all_dfs.append(df)
                else:
                    logger.warning(f"File {file_name} is missing 'text' or 'label' column.")
            except Exception as e:
                logger.error(f"Failed to read {file_name}: {e}")

    if not all_dfs:
        logger.warning(f"No valid datasets found in '{data_dir}'.")
        return pd.DataFrame(columns=["text", "label", "dataset_source"])

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Drop rows where 'text' or 'label' are NaN
    combined_df = combined_df.dropna(subset=['text', 'label'])
    
    # Ensure label is int
    combined_df['label'] = combined_df['label'].astype(int)
    
    logger.info(f"Harmonized {len(all_dfs)} datasets into {len(combined_df)} total records.")
    return combined_df
