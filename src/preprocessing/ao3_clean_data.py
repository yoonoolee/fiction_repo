"""
Data cleaning and preprocessing script
Cleans scraped stories and prepares them for fine-tuning
"""
import json
import re
from pathlib import Path
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    IDEAL_STORY_LENGTH_MIN,
    IDEAL_STORY_LENGTH_MAX,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT
)


class DataCleaner:
    """Clean and preprocess story data"""

    def __init__(self, input_file):
        """Initialize cleaner with input file"""
        self.input_file = input_file
        self.stories = []
        self.cleaned_stories = []

    def load_data(self):
        """Load stories from JSONL file"""
        print(f" Loading data from {self.input_file}...")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.stories.append(json.loads(line))

        print(f"   Loaded {len(self.stories)} stories")

    def count_words(self, text):
        """Count words in text"""
        return len(text.split())

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)

        # Remove common artifacts
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)

        # Fix common formatting issues
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def is_quality_story(self, story):
        """Check if story meets quality criteria"""
        text = story["text"]

        # Check word count (prefer ideal range)
        word_count = self.count_words(text)
        if word_count < IDEAL_STORY_LENGTH_MIN or word_count > IDEAL_STORY_LENGTH_MAX:
            return False

        # Check for minimum length after cleaning
        if len(text.strip()) < 100:  # At least 100 characters
            return False

        # Check for common spam patterns
        spam_patterns = [
            r'subscribe.*channel',
            r'like.*comment',
            r'click.*link',
            r'www\.',
            r'\.com',
        ]

        for pattern in spam_patterns:
            if re.search(pattern, text.lower()):
                return False

        # Check for excessive repetition (likely spam)
        words = text.lower().split()
        if len(words) > 10:
            # Count most common word
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)[0]
            # If a word appears more than 30% of the time, likely spam
            if most_common[1] / len(words) > 0.3:
                return False

        # For Reddit, check score (optional quality filter)
        if story.get("source") == "reddit":
            # Prefer stories with at least some engagement
            if story.get("score", 0) < 2:
                return False

        # For AO3, check kudos (optional quality filter)
        if story.get("source") == "ao3":
            # Prefer stories with at least some kudos
            if story.get("kudos", 0) < 5:
                return False

        return True

    def process_stories(self):
        """Clean and filter stories"""
        print("\n Cleaning and filtering stories...")

        for story in self.stories:
            # Clean text
            cleaned_text = self.clean_text(story["text"])

            # Update story with cleaned text
            story["text"] = cleaned_text
            story["word_count"] = self.count_words(cleaned_text)

            # Check quality
            if self.is_quality_story(story):
                self.cleaned_stories.append(story)

        print(f"   Kept {len(self.cleaned_stories)} / {len(self.stories)} stories")
        print(f"   Filtered out {len(self.stories) - len(self.cleaned_stories)} low-quality stories")

    def split_data(self):
        """Split data into train/val/test sets"""
        print("\n Splitting data into train/val/test...")

        import random
        random.seed(42)  # For reproducibility

        # Shuffle stories
        stories = self.cleaned_stories.copy()
        random.shuffle(stories)

        # Calculate split points
        total = len(stories)
        train_end = int(total * TRAIN_SPLIT)
        val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

        train_data = stories[:train_end]
        val_data = stories[train_end:val_end]
        test_data = stories[val_end:]

        print(f"   Train: {len(train_data)} stories ({TRAIN_SPLIT*100:.0f}%)")
        print(f"   Val:   {len(val_data)} stories ({VAL_SPLIT*100:.0f}%)")
        print(f"   Test:  {len(test_data)} stories ({TEST_SPLIT*100:.0f}%)")

        return train_data, val_data, test_data

    def save_split(self, data, filename):
        """Save data split to JSONL file"""
        output_path = PROCESSED_DATA_DIR / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for story in data:
                f.write(json.dumps(story) + '\n')

        print(f"   Saved {filename}")
        return output_path

    def print_statistics(self, train_data, val_data, test_data):
        """Print dataset statistics"""
        print("\n" + "=" * 60)
        print(" Dataset Statistics")
        print("=" * 60)

        all_data = train_data + val_data + test_data

        # Word count statistics
        word_counts = [s["word_count"] for s in all_data]
        print(f"\nWord counts:")
        print(f"  Average: {sum(word_counts) / len(word_counts):.1f}")
        print(f"  Min: {min(word_counts)}")
        print(f"  Max: {max(word_counts)}")

        # Source distribution
        sources = Counter([s["source"] for s in all_data])
        print(f"\nSource distribution:")
        for source, count in sources.items():
            print(f"  {source}: {count} ({count/len(all_data)*100:.1f}%)")

        # Reddit subreddit distribution
        reddit_stories = [s for s in all_data if s.get("source") == "reddit"]
        if reddit_stories:
            subreddits = Counter([s.get("subreddit", "unknown") for s in reddit_stories])
            print(f"\nReddit subreddit distribution:")
            for subreddit, count in subreddits.most_common(5):
                print(f"  r/{subreddit}: {count}")

        # Sample stories
        print(f"\n" + "=" * 60)
        print(" Sample Stories")
        print("=" * 60)

        for i, story in enumerate(all_data[:3], 1):
            print(f"\nSample {i} ({story['source']}):")
            print(f"Words: {story['word_count']}")
            print(f"Text: {story['text'][:200]}...")
            print("-" * 60)


def main():
    """Main cleaning function"""
    print("=" * 60)
    print("Data Cleaning and Preprocessing")
    print("=" * 60)

    # Input file (combined dataset)
    input_file = RAW_DATA_DIR / "combined_stories.jsonl"

    if not input_file.exists():
        print(f"\n Error: {input_file} not found")
        print("   Please run data scraping first: python src/scraping/run_all_scrapers.py")
        return

    # Initialize cleaner
    cleaner = DataCleaner(input_file)

    # Load data
    cleaner.load_data()

    # Process stories
    cleaner.process_stories()

    # Split data
    train_data, val_data, test_data = cleaner.split_data()

    # Save splits
    print("\n Saving processed data...")
    cleaner.save_split(train_data, "train.jsonl")
    cleaner.save_split(val_data, "val.jsonl")
    cleaner.save_split(test_data, "test.jsonl")

    # Print statistics
    cleaner.print_statistics(train_data, val_data, test_data)

    print("\n" + "=" * 60)
    print(" Data preprocessing complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Create template libraries: python src/preprocessing/create_templates.py")
    print("  2. Fine-tune model: notebooks/fine_tune_llama.ipynb")


if __name__ == "__main__":
    main()
