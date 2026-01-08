"""
Download Reddit Confessions dataset from HuggingFace
"""
# imports
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR


def download_confession_dataset():
    """
    Download Confession dataset from HuggingFace (streaming mode).
    """
    print("Downloading r/confession dataset with filters (streaming mode)...")
    # Use streaming mode to filter before downloading everything
    confession_dataset = load_dataset("SocialGrep/one-million-reddit-confessions", streaming=True)
    return confession_dataset


def convert_confession_format(dataset_stream):
    """
    Convert Confession dataset format to standardized dict format with filters.
    Uses streaming mode to filter during iteration without downloading everything.

    Args:
        dataset_stream: Streaming dataset from HuggingFace
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

        # Filter 2: Check score (must be between 100-400)
        score = item.get('score', 0)
        if score < 100 or score > 400:
            filtered_count += 1
            continue

        # Get text fields
        story_text = item.get('body', '') or item.get('selftext', '') or item.get('text', '')
        title = item.get('title', '')

        # Filter 3: exclude empty stories and removed/deleted content
        if not story_text or story_text in ['[removed]', '[deleted]']:
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


def save_confession_dataset(stories_df):
    """
    Save the Confession dataset
    """
    output_path = RAW_DATA_DIR / "reddit_confession.parquet"
    stories_df.to_parquet(output_path, index=False)

    print(f"\n=== Confession Dataset Summary ===")
    print(f"Total stories: {len(stories_df)}")
    print(f"Saved to: {output_path}")


def main():
    # Download dataset
    confession_dataset = download_confession_dataset()

    # Convert and filter
    stories = convert_confession_format(confession_dataset)
    stories_df = pd.DataFrame(stories)

    # Save
    save_confession_dataset(stories_df)
    print("\nReddit Confession stories saved successfully!")


if __name__ == "__main__":
    main()
