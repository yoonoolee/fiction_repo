"""
Download Reddit WritingPrompts datasets from HuggingFace
"""
# imports 
from datasets import load_dataset
import pandas as pd 
from pathlib import Path
import sys
import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR


def get_datasets(): 
    """
    Download reddit stories datasets from HuggingFace. 
    """
    # TODO: add more datasets? only 1 right now 
    wp_dataset = load_dataset("euclaise/WritingPrompts_curated") 

    return [wp_dataset]

def convert_format(dataset, subreddit='', nsfw=False):
    """
    Convert original input dataset format to dict format
    """
    stories = []

    for i, item in enumerate(tqdm.tqdm(dataset['train'], desc=subreddit)):
        story_text = item.get('body', '') or item.get('story', '') or item.get('text', '') or item.get('response', '')
        
        # exclude empty stories 
        if not story_text:
            continue
        
        word_count = len(story_text.split())
        
        story = {
            "id": f"{subreddit}_{i}",
            "text": story_text,
            "title": item.get('prompt', '') or item.get('title', ''),
            "word_count": word_count,
            "source": 'r/' + subreddit,
            "nsfw": nsfw
        }
        
        stories.append(story)
    
    return stories 

def convert_datasets_to_df(datasets): 
    """
    Convert and combine all datasets into pandas DataFrame
    """
    all_stories = []

    for dataset in datasets: 
        all_stories.extend(convert_format(dataset, subreddit='WritingPrompts', nsfw=False))

    all_stories = pd.DataFrame(all_stories)
    
    return all_stories 

def save_datasets(combined_df):
    """
    Save the data
    """

    combined_df.to_csv(RAW_DATA_DIR / "reddit_stories.csv", index=False)

def main():
    datasets = get_datasets()
    df = convert_datasets_to_df(datasets)
    save_datasets(df)
    print("Reddit stories from HuggingFace saved successfully!")


if __name__ == "__main__":
    main()
