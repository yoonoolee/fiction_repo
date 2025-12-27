"""
Download Reddit TIFU and Confession datasets
"""
# imports
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import zipfile
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR


def get_datasets():
    """
    Download TIFU (from Kaggle) and Confession datasets.
    """
    print("Downloading r/tifu dataset from Kaggle...")
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download TIFU dataset from Kaggle
    temp_dir = RAW_DATA_DIR / "temp_tifu"
    temp_dir.mkdir(exist_ok=True, parents=True)

    api.dataset_download_files('sanme1/reddit-tifu', path=str(temp_dir), unzip=True)

    # Load the CSV file
    tifu_files = list(temp_dir.glob('*.csv'))
    if not tifu_files:
        raise FileNotFoundError(f"No CSV files found in {temp_dir}")

    tifu_df = pd.read_csv(tifu_files[0])
    print(f"Loaded TIFU dataset with {len(tifu_df)} rows")

    print("\nDownloading r/confession dataset with filters (streaming mode)...")
    # Use streaming mode to filter before downloading everything
    confession_dataset = load_dataset("SocialGrep/one-million-reddit-confessions", streaming=True)

    return [
        (tifu_df, 'tifu'),
        (confession_dataset, 'confession')
    ]


def convert_tifu_format(df, subreddit='tifu'):
    """
    Convert TIFU dataset (from Kaggle CSV) to standardized dict format with filters
    """
    stories = []

    # Define filter criteria (same banned words as confessions)
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

    print(f"Processing {subreddit} dataset...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {subreddit}"):
        # Get values from CSV (keeping only specified columns)
        title = row.get('title', '')
        url = row.get('url', '')
        num_comments = row.get('num_comments', 0)
        tldr = row.get('tldr', '')
        story_text = row.get('selftext', '')
        upvote_ratio = row.get('upvote_ratio', 0)
        score = row.get('score', 0)
        trimmed_title = row.get('trimmed_title', '')
        ups = row.get('ups', 0)

        # Apply filters
        if upvote_ratio <= 0.7:
            filtered_count += 1
            continue
        if score <= 100:
            filtered_count += 1
            continue
        if num_comments <= 30:
            filtered_count += 1
            continue
        if not story_text or pd.isna(story_text):
            filtered_count += 1
            continue
        if contains_banned_words(story_text) or contains_banned_words(title) or contains_banned_words(tldr):
            filtered_count += 1
            continue

        # Keep story with 0-indexed ID (all requested columns)
        story = {
            "id": f"tifu_{kept_count}",
            "text": str(story_text),  # Standardized column name (was selftext)
            "title": str(title),
            "url": str(url) if not pd.isna(url) else '',
            "num_comments": int(num_comments),
            "tldr": str(tldr) if not pd.isna(tldr) else '',
            "upvote_ratio": float(upvote_ratio),
            "score": int(score),
            "trimmed_title": str(trimmed_title) if not pd.isna(trimmed_title) else '',
            "ups": int(ups),
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


def convert_confession_format(dataset_stream, subreddit='confession'):
    """
    Convert Confession dataset format to standardized dict format with filters.
    Uses streaming mode to filter during iteration without downloading everything.

    Args:
        dataset_stream: Streaming dataset from HuggingFace
        subreddit: Name of subreddit
    """
    stories = []

    # Define filter criteria
    ALLOWED_DOMAINS = ['self.confession', 'self.confessions']
    BANNED_WORDS = [
        'suicide', 'kidnap', 'kidnapped', 'kidnapping', 'cheating', 'minors',
        'aborted', 'abortion', 'racism', 'racist', 'rape', 'raping', 'raped',
        'rapist', 'molested', 'debt', 'covid', 'covid-19', 'covid 19', 'trump',
        'therapy', 'medication', 'trans', 'transphobic', 'ugly'
    ]

    def contains_banned_words(text):
        """Check if text contains any banned words (case insensitive)"""
        if not text:
            return False
        text_lower = text.lower()
        return any(banned_word in text_lower for banned_word in BANNED_WORDS)

    filtered_count = 0
    total_count = 0
    kept_count = 0  # 0-indexed counter for stories that pass filters

    print(f"Streaming and filtering confession posts...")

    # Iterate through streaming dataset - only downloads what we iterate through
    for item in dataset_stream['train']:
        total_count += 1

        # Filter 1: Check domain
        domain = item.get('domain', '')
        if domain not in ALLOWED_DOMAINS:
            filtered_count += 1
            continue

        # Filter 2: Check score (must be > 700)
        score = item.get('score', 0)
        if score <= 700:
            filtered_count += 1
            continue

        # Get text fields
        story_text = item.get('body', '') or item.get('selftext', '') or item.get('text', '')
        title = item.get('title', '')

        # Filter 3: exclude empty stories
        if not story_text:
            filtered_count += 1
            continue

        # Filter 4: Check for banned words in title and selftext
        if contains_banned_words(title) or contains_banned_words(story_text):
            filtered_count += 1
            continue

        # Keep story with 0-indexed ID
        story = {
            "id": f"confessions_{kept_count}",
            "text": story_text,  # Standardized column name
            "title": title,
            "score": score,
            "permalink": item.get('permalink', ''),
            "domain": domain,
            "source": "reddit_confession"
        }

        stories.append(story)
        kept_count += 1

        # Progress update every 1000 processed items
        if total_count % 1000 == 0:
            print(f"Processed {total_count} items, kept {kept_count} stories...", end='\r')

    print(f"\n\nFiltered out {filtered_count}/{total_count} posts ({filtered_count/total_count*100:.1f}%)")
    print(f"Kept {len(stories)} posts")

    return stories


def convert_datasets_to_df(datasets):
    """
    Convert and combine all datasets into pandas DataFrame
    """
    all_stories = []

    for dataset, name in datasets:
        if name == 'tifu':
            all_stories.extend(convert_tifu_format(dataset, subreddit='tifu'))
        elif name == 'confession':
            all_stories.extend(convert_confession_format(dataset, subreddit='confession'))

    all_stories_df = pd.DataFrame(all_stories)

    return all_stories_df


def save_datasets(combined_df):
    """
    Save the combined dataset only
    """
    combined_df.to_csv(RAW_DATA_DIR / "reddit_stories.csv", index=False)

    # Count stories by source (based on ID prefix)
    tifu_count = combined_df['id'].str.startswith('tifu_').sum()
    confession_count = combined_df['id'].str.startswith('confessions_').sum()

    print(f"\n=== Dataset Summary ===")
    print(f"Total stories: {len(combined_df)}")
    print(f"TIFU stories: {tifu_count}")
    print(f"Confession stories: {confession_count}")
    print(f"Saved to: {RAW_DATA_DIR / 'reddit_stories.csv'}")


def main():
    datasets = get_datasets()
    df = convert_datasets_to_df(datasets)
    save_datasets(df)
    print("\nReddit TIFU and Confession stories saved successfully!")


if __name__ == "__main__":
    main()
