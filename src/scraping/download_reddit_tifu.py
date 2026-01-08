"""
Download Reddit TIFU dataset from Kaggle
"""
# imports
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR


def download_tifu_dataset():
    """
    Download TIFU dataset from Kaggle.
    """
    print("Downloading r/tifu dataset from Kaggle...")
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download TIFU dataset from Kaggle
    temp_dir = RAW_DATA_DIR / "temp_tifu"
    temp_dir.mkdir(exist_ok=True, parents=True)

    api.dataset_download_files('sanme1/reddit-tifu', path=str(temp_dir), unzip=True)

    # Load the JSON file (it's JSON Lines format - one JSON object per line)
    tifu_files = list(temp_dir.glob('*.json'))
    if not tifu_files:
        raise FileNotFoundError(f"No JSON files found in {temp_dir}")

    # Read JSON Lines file
    tifu_data = []
    with open(tifu_files[0], 'r') as f:
        for line in f:
            tifu_data.append(json.loads(line))

    tifu_df = pd.DataFrame(tifu_data)
    print(f"Loaded TIFU dataset with {len(tifu_df)} rows")

    return tifu_df


def convert_tifu_format(df):
    """
    Convert TIFU dataset to standardized dict format with filters
    """
    stories = []

    # Define filter criteria
    BANNED_WORDS = [
        'suicide', 'kidnap', 'kidnapped', 'kidnapping', 'cheating', 'minors',
        'aborted', 'abortion', 'racism', 'racist', 'rape', 'raping', 'raped',
        'rapist', 'molested', 'debt', 'covid', 'covid-19', 'covid 19', 'trump',
        'therapy', 'medication', 'trans', 'transphobic', 'ugly'
    ]

    def contains_banned_words(text):
        """Check if text contains any banned words (case insensitive)"""
        if not text or pd.isna(text):
            return False
        text_lower = str(text).lower()
        return any(banned_word in text_lower for banned_word in BANNED_WORDS)

    filtered_count = 0
    total_count = len(df)
    kept_count = 0

    print(f"Processing tifu dataset...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing tifu"):
        # Get values from CSV
        title = row.get('title', '')
        url = row.get('url', '')
        tldr = row.get('tldr', '')
        story_text = row.get('selftext', '')
        upvote_ratio = row.get('upvote_ratio', 0)
        score = row.get('score', 0)
        trimmed_title = row.get('trimmed_title', '')
        ups = row.get('ups', 0)

        # Apply filters
        if pd.isna(tldr) or len(tldr) < 5:
            filtered_count += 1
            continue
        if score < 100:
            filtered_count += 1
            continue
        if not story_text or pd.isna(story_text):
            filtered_count += 1
            continue
        if contains_banned_words(story_text) or contains_banned_words(title) or contains_banned_words(tldr):
            filtered_count += 1
            continue

        # Keep story with 0-indexed ID
        story = {
            "id": f"tifu_{kept_count}",
            "title": str(trimmed_title) if not pd.isna(trimmed_title) else '',
            "text": str(tldr) if not pd.isna(tldr) else '',
            "url": str(url) if not pd.isna(url) else '',
            "score": int(score),
            "source": "reddit_tifu"
        }

        stories.append(story)
        kept_count += 1

    print(f"\nFiltered out {filtered_count}/{total_count} posts ({filtered_count/total_count*100:.1f}%)")
    print(f"Kept {len(stories)} posts")

    # Clean up temp directory
    temp_dir = RAW_DATA_DIR / "temp_tifu"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)

    return stories


def save_tifu_dataset(stories_df):
    """
    Save the TIFU dataset
    """
    output_path = RAW_DATA_DIR / "reddit_tifu.parquet"
    stories_df.to_parquet(output_path, index=False)

    print(f"\n=== TIFU Dataset Summary ===")
    print(f"Total stories: {len(stories_df)}")
    print(f"Saved to: {output_path}")


def main():
    # Download dataset
    tifu_df = download_tifu_dataset()

    # Convert and filter
    stories = convert_tifu_format(tifu_df)
    stories_df = pd.DataFrame(stories)

    # Save
    save_tifu_dataset(stories_df)
    print("\nReddit TIFU stories saved successfully!")


if __name__ == "__main__":
    main()
