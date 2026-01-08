"""
Process AO3 and TIFU datasets with emotion, tone, and genre classification
"""
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
from transformers import pipeline
import json
import shutil
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR

PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Detect device (MPS for Mac GPU, fallback to CPU)
if torch.backends.mps.is_available():
    DEVICE = 0  # Use GPU
    print("✓ Using Mac GPU (MPS) for acceleration")
else:
    DEVICE = -1  # CPU
    print("⚠ MPS not available, using CPU (slower)")


def load_datasets():
    """Load AO3 and TIFU datasets"""
    print("Loading datasets...")

    # Load AO3
    ao3_path = RAW_DATA_DIR / "ao3_fragments_selfcontained.parquet"
    ao3_df = pd.read_parquet(ao3_path)
    print(f"Loaded {len(ao3_df)} AO3 fragments")

    # Load TIFU
    tifu_path = RAW_DATA_DIR / "reddit_tifu.parquet"
    tifu_df = pd.read_parquet(tifu_path)
    print(f"Loaded {len(tifu_df)} TIFU stories")

    return ao3_df, tifu_df


def prepare_ao3_dataset(df):
    """Keep only needed columns and add type"""
    print("\nPreparing AO3 dataset...")
    df = df[['id', 'text', 'emotion', 'emotion_score']].copy()
    df['type'] = 'short_story'
    print(f"AO3 dataset prepared: {len(df)} rows")
    return df


def prepare_tifu_dataset(df):
    """Keep only needed columns and add type"""
    print("\nPreparing TIFU dataset...")
    df = df[['id', 'text']].copy()
    df['type'] = 'one_liner'
    print(f"TIFU dataset prepared: {len(df)} rows")
    return df


def classify_emotion_tifu(df, checkpoint_file):
    """Classify emotions for TIFU dataset and filter"""
    print("\n=== Classifying TIFU Emotions ===")
    print("Loading emotion classifier...")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=DEVICE
    )

    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        df_checkpoint = pd.DataFrame(checkpoint_data['results'])
        start_idx = checkpoint_data['last_processed'] + 1
        print(f"Resuming from row {start_idx}")
    else:
        df_checkpoint = pd.DataFrame()

    results = []

    try:
        for idx in tqdm(range(start_idx, len(df)), desc="Classifying emotions"):
            text = df.iloc[idx]['text']

            # Classify
            predictions = emotion_classifier(str(text)[:512])[0]  # Limit to 512 chars

            # Get highest scoring emotion
            top_emotion = max(predictions, key=lambda x: x['score'])

            result = {
                'id': df.iloc[idx]['id'],
                'text': df.iloc[idx]['text'],
                'type': df.iloc[idx]['type'],
                'emotion': top_emotion['label'],
                'emotion_score': top_emotion['score']
            }
            results.append(result)

            # Save checkpoint every 100 rows
            if (idx + 1) % 100 == 0:
                checkpoint_data = {
                    'last_processed': idx,
                    'results': results
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_data = {
            'last_processed': idx - 1,
            'results': results
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"Checkpoint saved at row {idx}")
        raise

    # Combine with checkpoint data
    if not df_checkpoint.empty:
        results_df = pd.concat([df_checkpoint, pd.DataFrame(results)], ignore_index=True)
    else:
        results_df = pd.DataFrame(results)

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return results_df


def classify_tone(df, checkpoint_file):
    """Classify tone for both datasets using go_emotions - only keeps top tone if score >= 0.8"""
    print("\n=== Classifying Tone (Go Emotions) ===")
    print("Loading tone classifier...")
    tone_classifier = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device=DEVICE
    )

    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        df = pd.DataFrame(checkpoint_data['df'])
        start_idx = checkpoint_data['last_processed'] + 1
        print(f"Resuming from row {start_idx}")

    df['tone'] = None
    df['tone_score'] = None

    try:
        for idx in tqdm(range(start_idx, len(df)), desc="Classifying tone"):
            text = df.iloc[idx]['text']

            # Classify
            predictions = tone_classifier(str(text)[:2000])[0]

            # Get top tone
            top_tone = max(predictions, key=lambda x: x['score'])

            # Only keep if score >= 0.8
            if top_tone['score'] >= 0.8:
                df.at[idx, 'tone'] = top_tone['label']
                df.at[idx, 'tone_score'] = round(top_tone['score'], 3)

            # Save checkpoint every 100 rows
            if (idx + 1) % 100 == 0:
                checkpoint_data = {
                    'last_processed': idx,
                    'df': df.to_dict('records')
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_data = {
            'last_processed': idx - 1,
            'df': df.to_dict('records')
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"Checkpoint saved at row {idx}")
        raise

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return df


def classify_genre(df, checkpoint_file):
    """Classify genre using jquigl/electra-movie-genre"""
    print("\n=== Classifying Genre ===")
    print("Loading genre classifier...")
    from transformers import AutoTokenizer, ElectraForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("deepset/electra-base-squad2")
    model = ElectraForSequenceClassification.from_pretrained('jquigl/electra-movie-genre')
    genre_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=DEVICE)

    # Genre mapping
    GENRE_LABELS = {
        'LABEL_0': 'Fantasy', 'LABEL_1': 'Romance', 'LABEL_2': 'Thriller',
        'LABEL_3': 'Biography', 'LABEL_4': 'Horror', 'LABEL_5': 'Action',
        'LABEL_6': 'Crime', 'LABEL_7': 'Animation', 'LABEL_8': 'Adventure',
        'LABEL_9': 'Mystery', 'LABEL_10': 'War', 'LABEL_11': 'Family',
        'LABEL_12': 'History', 'LABEL_13': 'Scifi', 'LABEL_14': 'Film-noir',
        'LABEL_15': 'Sports'
    }

    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        df = pd.DataFrame(checkpoint_data['df'])
        start_idx = checkpoint_data['last_processed'] + 1
        print(f"Resuming from row {start_idx}")

    df['genre'] = None
    df['genre_score'] = None

    try:
        for idx in tqdm(range(start_idx, len(df)), desc="Classifying genre"):
            text = df.iloc[idx]['text']

            # Classify
            result = genre_classifier(str(text)[:512])[0]

            # Map label to genre name
            genre = GENRE_LABELS.get(result['label'], result['label'])
            score = result['score']

            df.at[idx, 'genre'] = genre
            df.at[idx, 'genre_score'] = score

            # Save checkpoint every 100 rows
            if (idx + 1) % 100 == 0:
                checkpoint_data = {
                    'last_processed': idx,
                    'df': df.to_dict('records')
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_data = {
            'last_processed': idx - 1,
            'df': df.to_dict('records')
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"Checkpoint saved at row {idx}")
        raise

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return df


def main():
    print("="*80)
    print("Story Processing Pipeline")
    print("="*80)

    checkpoint_dir = PROCESSED_DATA_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Define intermediate save paths
    after_emotion_pq = PROCESSED_DATA_DIR / "_intermediate_after_emotion.parquet"
    after_tone_pq = PROCESSED_DATA_DIR / "_intermediate_after_tone.parquet"
    after_genre_pq = PROCESSED_DATA_DIR / "_intermediate_after_genre.parquet"

    # Check where to resume from
    if after_genre_pq.exists():
        print("\n✓ Found completed genre classification, loading...")
        combined_df = pd.read_parquet(after_genre_pq)
        print(f"Loaded {len(combined_df)} stories from after genre step")
    elif after_tone_pq.exists():
        print("\n⟳ Resuming from tone step...")
        combined_df = pd.read_parquet(after_tone_pq)
        print(f"Loaded {len(combined_df)} stories")

        genre_checkpoint = checkpoint_dir / "genre_checkpoint.json"
        combined_df = classify_genre(combined_df, genre_checkpoint)
        combined_df.to_parquet(after_genre_pq, index=False)
        after_tone_pq.unlink()
        print(f"✓ Saved progress after genre classification")
    elif after_emotion_pq.exists():
        print("\n⟳ Resuming from emotion step...")
        combined_df = pd.read_parquet(after_emotion_pq)
        print(f"Loaded {len(combined_df)} stories")

        tone_checkpoint = checkpoint_dir / "tone_checkpoint.json"
        combined_df = classify_tone(combined_df, tone_checkpoint)
        combined_df.to_parquet(after_tone_pq, index=False)
        after_emotion_pq.unlink()
        print(f"✓ Saved progress after tone classification")

        genre_checkpoint = checkpoint_dir / "genre_checkpoint.json"
        combined_df = classify_genre(combined_df, genre_checkpoint)
        combined_df.to_parquet(after_genre_pq, index=False)
        after_tone_pq.unlink()
        print(f"✓ Saved progress after genre classification")
    else:
        print("\n⟳ Starting from beginning...")

        # Load datasets
        ao3_df, tifu_df = load_datasets()

        # Prepare datasets
        ao3_df = prepare_ao3_dataset(ao3_df)
        tifu_df = prepare_tifu_dataset(tifu_df)

        # Process TIFU: Add emotion classification
        tifu_emotion_checkpoint = checkpoint_dir / "tifu_emotion_checkpoint.json"
        tifu_df = classify_emotion_tifu(tifu_df, tifu_emotion_checkpoint)

        # Combine datasets
        print(f"\n=== Combining Datasets ===")
        print(f"AO3: {len(ao3_df)} stories")
        print(f"TIFU: {len(tifu_df)} stories")
        combined_df = pd.concat([ao3_df, tifu_df], ignore_index=True)
        print(f"Combined: {len(combined_df)} stories")
        combined_df.to_parquet(after_emotion_pq, index=False)
        print(f"✓ Saved progress after emotion classification")

        # Process both: Tone classification
        tone_checkpoint = checkpoint_dir / "tone_checkpoint.json"
        combined_df = classify_tone(combined_df, tone_checkpoint)
        combined_df.to_parquet(after_tone_pq, index=False)
        after_emotion_pq.unlink()
        print(f"✓ Saved progress after tone classification")

        # Process both: Genre classification
        genre_checkpoint = checkpoint_dir / "genre_checkpoint.json"
        combined_df = classify_genre(combined_df, genre_checkpoint)
        combined_df.to_parquet(after_genre_pq, index=False)
        after_tone_pq.unlink()
        print(f"✓ Saved progress after genre classification")

    # Save final dataset
    output_path = PROCESSED_DATA_DIR / "ao3_tifu_enriched_labels.parquet"
    combined_df.to_parquet(output_path, index=False)

    # Clean up all intermediate files and checkpoints
    print("\n🧹 Cleaning up intermediate files...")
    if after_genre_pq.exists():
        after_genre_pq.unlink()

    # Clean up any remaining checkpoint files
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print("✓ Removed checkpoint directory")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nFinal dataset: {len(combined_df)} stories")
    print(f"Saved to: {output_path}")
    print(f"\nColumns: {list(combined_df.columns)}")

    # Show summary stats
    print(f"\n=== Summary ===")
    print(f"By type:")
    print(combined_df['type'].value_counts())
    print(f"\nBy emotion:")
    print(combined_df['emotion'].value_counts())
    print(f"\nBy tone (only scores >= 0.8):")
    print(combined_df['tone'].value_counts().head(10))
    print(f"\nBy genre:")
    print(combined_df['genre'].value_counts().head(10))


if __name__ == "__main__":
    main()
