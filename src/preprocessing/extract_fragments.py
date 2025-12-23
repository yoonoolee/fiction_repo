"""
Extract emotionally intense 40-60 word fragments from AO3 stories

This script:
1. Loads scraped AO3 stories from CSV
2. Extracts 10-40 word sliding windows
3. Uses emotion classification to find intense moments
4. Filters for narrative content
5. Saves processed fragments for training
"""

import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import sys
import spacy
import nltk
import time
from transformers import pipeline

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Load emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Load NSFW classifier
nsfw_classifier = pipeline(
    "text-classification",
    model="michellejieli/NSFW_text_classifier"
)

# Download NLTK sentence tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

# spaCy for NER/POS on fragments
nlp = spacy.load("en_core_web_sm")

class FragmentExtractor:
    def __init__(self, min_words=10, max_words=40, emotion_threshold=0.8, batch_size=32):
        self.min_words = min_words
        self.max_words = max_words
        self.emotion_threshold = emotion_threshold
        self.batch_size = batch_size

    def extract_sentence_groups(self, text):
        """Extract 2-4 complete sentences using NLTK (fast sentence tokenization)"""
        sentences = nltk.sent_tokenize(text)
        
        fragments = []
        
        # Try groups of 2, 3, and 4 consecutive sentences
        for group_size in [2, 3, 4]:
            for i in range(len(sentences) - group_size + 1):
                group = sentences[i:i + group_size]
                combined = ' '.join(group)
                
                word_count = len(combined.split())
                if self.min_words <= word_count <= self.max_words: 
                    fragments.append(combined)
        
        return fragments

    def clean_fragment(self, text):
        """Remove extra whitespace, normalize spacing, and confirm complete sentence structure."""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        if text and text[-1] not in '.!?':
            text += '.'
        return text

    def has_narrative_content_batch(self, texts):
        """
        Batch check if fragments have person entity and verb.
        Uses spaCy's NER and POS tagging with batched processing.
        """
        results = []

        # Process all texts in batches
        for doc in nlp.pipe(texts, batch_size=self.batch_size):
            has_narrative = False
            for sent in doc.sents:
                has_person = any(ent.label_ == "PERSON" for ent in sent.ents)
                has_verb = any(token.pos_ == "VERB" for token in sent)

                if has_person and has_verb:
                    has_narrative = True
                    break

            results.append(has_narrative)

        return results

    def get_emotion_scores_batch(self, texts):
        """Get emotion scores for a batch of texts"""
        try:
            results = emotion_classifier(texts, batch_size=self.batch_size)
            emotions = []
            for result in results:
                best = max(result, key=lambda x: x['score'])
                emotions.append((best['label'], best['score']))
            return emotions
        except:
            return [(None, 0.0)] * len(texts)

    def get_nsfw_labels_batch(self, texts):
        """Get NSFW labels for a batch of texts"""
        try:
            results = nsfw_classifier(texts, batch_size=self.batch_size)
            return [result['label'] == 'NSFW' and result['score'] > 0.9 for result in results]
        except:
            return [False] * len(texts)

    def extract_from_story(self, story):
        text = story['text']

        candidates = self.extract_sentence_groups(text)

        # Step 1: Clean and deduplicate
        cleaned_fragments = []
        seen = set()
        for frag_text in candidates:
            clean_text = self.clean_fragment(frag_text)
            if clean_text not in seen:
                seen.add(clean_text)
                cleaned_fragments.append(clean_text)

        if not cleaned_fragments:
            return []

        # Step 2: Batch emotion classification
        t = time.time()
        emotion_results = self.get_emotion_scores_batch(cleaned_fragments)
        print(f"  Emotion classify: {time.time() - t:.2f}s")

        # Step 3: Filter by emotion threshold and neutral
        high_emotion_fragments = []
        high_emotion_data = []
        for frag_text, (emotion_label, emotion_score) in zip(cleaned_fragments, emotion_results):
            if emotion_score >= self.emotion_threshold and emotion_label != "neutral":
                high_emotion_fragments.append(frag_text)
                high_emotion_data.append((emotion_label, emotion_score))

        if not high_emotion_fragments:
            return []

        # Step 4: Batch check narrative content (spaCy)
        t = time.time()
        narrative_results = self.has_narrative_content_batch(high_emotion_fragments)
        print(f"  Narrative check: {time.time() - t:.2f}s")

        narrative_fragments = []
        narrative_data = []
        for frag_text, emotion_data, has_narrative in zip(high_emotion_fragments, high_emotion_data, narrative_results):
            if has_narrative:
                narrative_fragments.append(frag_text)
                narrative_data.append(emotion_data)

        if not narrative_fragments:
            return []

        # Step 5: Batch NSFW classification
        t = time.time()
        nsfw_results = self.get_nsfw_labels_batch(narrative_fragments)
        print(f"  NSFW classify: {time.time() - t:.2f}s")

        # Step 6: Assemble final fragments
        fragments = []
        for idx, (frag_text, (emotion_label, emotion_score), nsfw) in enumerate(
            zip(narrative_fragments, narrative_data, nsfw_results)
        ):
            fragments.append({
                'id': f"{story.get('id')}_{idx}",
                'text': frag_text,
                'word_count': len(frag_text.split()),
                'emotion': emotion_label,
                'emotion_score': round(emotion_score, 3),
                'nsfw': nsfw,
                'source_story_id': story['id'],
                'title': story.get('title', 'Unknown'),
                'tag': story.get('search_tag', ''),
                'tags': story.get('tags', []),
                'source': 'ao3',
                'kudos': story.get('kudos', 0),
                'url': story.get('url', '')
            })

        return fragments


def main():
    """Main extraction pipeline"""
    input_path = RAW_DATA_DIR / "ao3_stories.csv"
    print(f'Reading input from {input_path}...')
    df = pd.read_csv(input_path)
    extractor = FragmentExtractor(min_words=10, max_words=40, emotion_threshold=0.8)

    all_fragments = []
    batch_fragments = []
    batch_num = 1

    # extract fragments
    stories = df.to_dict('records')
    for i, story in enumerate(tqdm(stories, desc="Extracting fragments"), 1):
        if pd.isna(story.get('text')):
            continue
        print(f"[{i}/{len(stories)}] Processing: {story.get('title', 'Unknown')}...")
        frags = extractor.extract_from_story(story)
        all_fragments.extend(frags)
        batch_fragments.extend(frags)

        # Save intermediate batch every 5 stories
        if i % 5 == 0:
            batch_output_path = PROCESSED_DATA_DIR / f"ao3_fragments_batch_{batch_num}.csv"
            batch_df = pd.DataFrame(batch_fragments)
            batch_df.to_csv(batch_output_path, index=False)
            print(f'  Saved batch {batch_num}: {len(batch_fragments)} fragments -> {batch_output_path}')
            batch_fragments = []
            batch_num += 1

    # Save any remaining fragments
    if batch_fragments:
        batch_output_path = PROCESSED_DATA_DIR / f"ao3_fragments_batch_{batch_num}.csv"
        batch_df = pd.DataFrame(batch_fragments)
        batch_df.to_csv(batch_output_path, index=False)
        print(f'  Saved batch {batch_num}: {len(batch_fragments)} fragments -> {batch_output_path}')

    print(f'\n{len(all_fragments)} fragments in final.')

    # save final combined file
    output_path = PROCESSED_DATA_DIR / "ao3_fragments.csv"
    fragment_df = pd.DataFrame(all_fragments)
    print('Saving final DF...')
    fragment_df.to_csv(output_path, index=False)
    print(f'Saved final DF successfully! {output_path}')

if __name__ == "__main__":
    main()
