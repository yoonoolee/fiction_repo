"""
Split data into train/val/test (80/10/10). 
Convert all text in data into JSONL format. 
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import json 
from sklearn.model_selection import train_test_split
from config import PROCESSED_DATA_DIR

def save_jsonl(df, path):
    """Save dataframe as JSONL with just text field"""
    with open(path, 'w') as f:
        for text in df['text']:
            f.write(json.dumps({"text": text}) + '\n')

def main(): 
    print('Reading input data')
    input_path = PROCESSED_DATA_DIR / "ao3_fragments_selfcontained.csv"
    df = pd.read_csv(input_path)

    # Split data into train/val/test: 80/10/10
    train_df, valtest_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(valtest_df, test_size=0.5, random_state=42)

    # Save CSV 
    print('Start saving CSV')
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print('Saved CSV successfully!')

    # Save JSONL 
    train_jsonl_path = PROCESSED_DATA_DIR / "train.jsonl"
    val_jsonl_path = PROCESSED_DATA_DIR / "val.jsonl"
    test_jsonl_path = PROCESSED_DATA_DIR / "test.jsonl"

    print('Start saving JSONL')
    save_jsonl(train_df, train_jsonl_path)
    save_jsonl(val_df, val_jsonl_path)
    save_jsonl(test_df, test_jsonl_path)
    print('Saved JSONL successfully!')

if __name__ == "__main__": 
    main()
