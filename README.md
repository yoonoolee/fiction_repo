# FriendFic - AI Story Generation System

A personalized daily story generation system that creates absurdist, humorous narratives for friend groups using fine-tuned LLMs and simple randomized prompting.

## Project Overview

This system generates unique, personalized short stories for friend groups every day. Stories are:
- Absurdist and humorous with unexpected plot twists
- **4 story types**: short stories (20-40 words), one-liners (10-20 words), script-style dialogue, or poems
- Feature **2-3 randomly selected friends** from the friend group
- Include a **randomly chosen theme/emotion/genre** (e.g., romance, chaos, mystery, nostalgia)
- **Occasional personalization** (20% of days): includes one specific detail about one friend (pet names, interests, siblings, partners)
- Variable length based on type

## Architecture

### ML Components

1. **Fine-tuned Story Generator**: Llama-3.1-8B with LoRA adapters trained on absurdist stories from Reddit TIFU + AO3 fragments
2. **Random Prompt Builder**: Randomly selects story type, characters, theme, and optional personalization detail
3. **User Data Store**: Simple database storing friend attributes (pets, interests, siblings, partners)
4. **Theme Library**: Small curated list of themes/emotions/genres to randomly choose from
5. **Evaluation Framework**: Perplexity, Self-BLEU, and semantic similarity metrics

### Tech Stack

- **ML Framework**: PyTorch, HuggingFace Transformers, PEFT
- **Database**: SQLite (dev), PostgreSQL (production) - stores friend data (pets, interests, siblings, partners)
- **Backend**: FastAPI
- **Demo UI**: Streamlit
- **Deployment**: iOS app (Swift/SwiftUI)

## Project Structure

```
FriendFic/
 data/
    raw/              # Scraped training data (Reddit TIFU, Confessions, AO3)
    processed/        # Cleaned and split datasets (fragments)
    test/             # Test dataset (tifu_ao3.csv)
 notebooks/            # Jupyter notebooks for experiments
 src/
    scraping/         # Web scraping scripts (AO3, Reddit)
    preprocessing/    # Data cleaning and fragment extraction
    training/         # Model fine-tuning scripts (LoRA)
    generation/      # Story generation pipeline
    evaluation/      # Metrics and evaluation
 models/              # Saved model weights and LoRA adapters
 database/            # SQLite database (friend data: pets, interests, siblings, partners)
 streamlit/           # Demo interface
 requirements.txt     # Python dependencies
 config.py            # Configuration settings
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the root directory:

```
# Weights & Biases (for experiment tracking)
WANDB_API_KEY=your_wandb_key

# HuggingFace (for model downloads and datasets)
HF_TOKEN=your_huggingface_token

# Optional: OpenAI/Anthropic for fallback generation
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 3. Download Training Data

```bash
# Download Reddit stories from HuggingFace (no API needed)
python src/scraping/download_reddit_datasets.py

# Or use the Jupyter notebook
jupyter notebook notebooks/download_reddit_datasets.ipynb
```

### 4. Fine-tune Model

Use the Colab notebook in `notebooks/fine_tune_llama.ipynb` or run locally:

```bash
python src/training/finetune_llama.py
```

### 5. Generate Stories with Fine-Tuned Model

```bash
python src/generation/story_generator.py
```

### 6. Run Demo (Coming Soon)

```bash
streamlit run streamlit/app.py
```

## Story Generation & Personalization

### System Evolution

**v1 (MVP - Current):** Pure random selection + deduplication
- Random theme, story type, characters each day
- Collect ratings and prompt performance data
- Prevent duplicate stories per group using vector similarity

**v2 (Future - After 1-2 months):** Preference learning
- Analyze collected rating data
- Build preference embeddings per group (learned "taste profile")
- Switch to weighted random selection based on what each group likes
- System improves over time automatically

### Daily Story Pipeline (v1 - MVP)

1. **Select Story Type** (random)
   - Short story (20-40 words)
   - One-liner (10-20 words)
   - Script-style dialogue
   - Poem

2. **Select Characters** (random)
   - Pick 2-3 friends from the friend group
   - Use their actual names

3. **Select Theme/Emotion/Genre** (random)
   - Small curated list: romance, chaos, mystery, adventure, nostalgia, revenge, comedy, drama, thriller, etc.

4. **Personalization** (20% of days)
   - Randomly decide: personalize or not?
   - If yes: pick ONE friend + ONE attribute
   - **Attribute types:**
     - Pet: "Manasa's cat Remy"
     - Interest: "Padma likes fantasy books"
     - Sibling: "Alex has a younger brother"
     - Partner: "Sam is dating Jordan"

5. **Generate Story**
   - Build detailed prompt with: type + characters + theme + optional personalization
   - Send to fine-tuned Llama-3.1-8B-Instruct (NOT few-shot, model already trained on absurdist stories)
   - Different prompt every day = different story every day
   - Return story

### Example Prompts

**Without Personalization:**
```
Create a one-liner (10-20 words) featuring Alex and Padma in a romance story.
```

**With Personalization:**
```
Create a short story (20-40 words) featuring Manasa and Sam in a chaos story.
Include that Manasa has a cat named Remy.
```

### Database Schema

**Backend: PostgreSQL with pgvector (via Supabase)**

```sql
-- Friend Groups
CREATE TABLE friend_groups (
    group_id UUID PRIMARY KEY,
    group_name TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Friends (per group)
CREATE TABLE friends (
    friend_id UUID PRIMARY KEY,
    group_id UUID REFERENCES friend_groups(group_id),
    name TEXT,
    pets JSONB,         -- [{"type": "cat", "name": "Remy"}]
    interests JSONB,    -- ["fantasy books", "hiking"]
    siblings JSONB,     -- ["younger brother"]
    partner TEXT
);

-- Generated Stories (with embeddings for deduplication)
CREATE TABLE generated_stories (
    story_id UUID PRIMARY KEY,
    group_id UUID REFERENCES friend_groups(group_id),

    -- Story content
    story_text TEXT,
    story_embedding VECTOR(384),  -- For similarity search

    -- Generation parameters
    story_type TEXT,              -- "one-liner", "short story", "dialogue", "poem"
    theme TEXT,
    characters JSONB,             -- ["Alex", "Sam"]
    personalization_used BOOLEAN,
    personalization_detail TEXT,

    -- User feedback
    rating INTEGER,               -- 1-5 stars (nullable until rated)

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_group_date (group_id, created_at),
    INDEX idx_embedding USING ivfflat (story_embedding vector_cosine_ops)
);

-- Prompt Performance (for future preference learning)
CREATE TABLE prompt_performance (
    prompt_id UUID PRIMARY KEY,
    group_id UUID REFERENCES friend_groups(group_id),

    -- Prompt details
    prompt_text TEXT,
    prompt_embedding VECTOR(384),

    -- Parameters
    story_type TEXT,
    theme TEXT,
    characters JSONB,
    personalization_used BOOLEAN,
    personalization_detail TEXT,

    -- Results (for learning)
    resulting_story_id UUID REFERENCES generated_stories(story_id),
    rating INTEGER,  -- Copied from story when rated

    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_group_rating (group_id, rating)
);

-- Group Preferences (v2 - not used in MVP, but schema ready)
CREATE TABLE group_preferences (
    group_id UUID PRIMARY KEY REFERENCES friend_groups(group_id),

    -- Learned preferences (built from ratings data)
    preference_embedding VECTOR(384),     -- Average of highly-rated stories
    preferred_themes JSONB,               -- {"chaos": 0.8, "romance": 0.2}
    preferred_story_types JSONB,          -- {"one-liner": 0.6, "poem": 0.9}

    -- Statistics
    total_ratings_count INTEGER DEFAULT 0,
    average_rating FLOAT,

    last_updated TIMESTAMP DEFAULT NOW()
);
```

### Theme Library (Example)

Simple list in code (can expand later):
- romance
- chaos
- mystery
- adventure
- nostalgia
- revenge
- comedy
- drama
- thriller
- horror
- sci-fi
- fantasy
- slice-of-life
- absurdist
- wholesome

### Mobile App Architecture

```
┌─────────────────────┐
│  iOS App            │
│  (Swift/SwiftUI)    │
│  - Display stories  │
│  - Rate stories     │
│  - Manage friends   │
└──────────┬──────────┘
           │ HTTPS API
           ▼
┌─────────────────────────┐
│  Backend Server         │
│  (FastAPI)              │
│  - Generate stories     │
│  - Check duplicates     │
│  - Store ratings        │
│  - Load fine-tuned model│
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐    ┌─────────────────┐
│  Supabase                │    │  Fine-tuned     │
│  (PostgreSQL + pgvector) │    │  Llama Model    │
│  - Friend data           │    │  (GPU instance) │
│  - Story embeddings      │    └─────────────────┘
│  - Ratings & prompts     │
└──────────────────────────┘
```

## Vector Database Use Cases

### 1. Story Deduplication (v1 - MVP)
**Problem:** LLM might generate similar stories even with different prompts

**Solution:** Store embeddings of all generated stories per group
- Before showing a story, check similarity to last 30 days of stories
- If similarity > 0.85, regenerate with different random parameters
- Ensures true variety for each friend group

**Implementation:**
```sql
-- Query similar stories for a group
SELECT story_id, story_text,
       1 - (story_embedding <=> query_embedding) AS similarity
FROM generated_stories
WHERE group_id = $1
  AND created_at > NOW() - INTERVAL '30 days'
ORDER BY story_embedding <=> query_embedding
LIMIT 5;
```

### 2. Preference Learning (v2 - Future)
**Problem:** How to learn what each group likes over time?

**Solution:** Build preference embeddings from rated stories
- After collecting 50+ ratings, calculate average embedding of highly-rated stories
- This creates a "taste profile" vector for each group
- Use it to weight random selection toward similar themes/types

**v1 (Now):** Just collect the data
- Store all prompts + ratings in `prompt_performance` table
- Don't act on it yet

**v2 (Later):** Analyze and apply
- Build `group_preferences` from `prompt_performance` data
- Switch from pure random to weighted random selection
- 80% exploit (use preferences), 20% explore (random)

## Evaluation Metrics

### Training-Time (During Fine-Tuning)

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **Validation Perplexity** | < 50 | Ensures fluent, coherent text |
| **Validation Loss** | Decreasing | Detects overfitting |

### Post-Training (Before Deployment)

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **Self-BLEU** | < 0.3 | Stories are different from each other |
| **Length Accuracy** | >80% in range | Stories match target length (10-20 or 20-40 words) |

### Production (v1 - After Launch)

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **User Rating** | > 3.5/5 | **PRIMARY METRIC.** Only thing that matters. |
| **Deduplication Hit Rate** | < 10% | % regenerated due to similarity. Ensures variety. |

### v2 Evaluation (Preference Learning)

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **User Rating (v2 vs v1)** | > 4.0/5 AND p < 0.05 | Only deploy v2 if statistically better than v1 (Independent Samples T-Test) |

**Priority:** User ratings > Deduplication rate > Training diagnostics

## Development Roadmap

### Phase 1: Data & Training ✅
- [x] Set up project structure
- [x] Scrape Reddit stories (TIFU, Confessions)
- [x] Scrape AO3 stories with absurdist tags
- [x] Extract emotionally intense fragments
- [x] Clean and preprocess data (106K train, 13K val, 13K test)
- [x] Create fine-tuning script
- [ ] **NEXT**: Fine-tune Llama-3.1-8B-Instruct with LoRA on full dataset

### Phase 2: MVP Backend (v1) - Pure Random + Deduplication
- [ ] Set up Supabase project (PostgreSQL + pgvector)
- [ ] Create database schema (all 5 tables including v2-ready `group_preferences`)
- [ ] Create theme/emotion/genre library (~15-20 themes)
- [ ] Build FastAPI backend:
  - Random prompt builder (type, theme, characters, optional personalization)
  - Story deduplication with vector similarity
  - Store generated stories + embeddings
  - Store prompt performance data (for future v2)
  - Rating endpoint (updates both `generated_stories` and `prompt_performance`)
- [ ] Deploy fine-tuned model (GPU instance)
- [ ] API endpoints:
  - `POST /generate_story` - Generate daily story for group
  - `POST /rate_story` - Submit rating (1-5 stars)
  - `GET /friend_groups` - List groups
  - `POST /friends` - Add/edit friend data

### Phase 3: iOS App (v1)
- [ ] Build SwiftUI app:
  - Friend group management
  - Friend profiles (name, pets, interests, siblings, partner)
  - Daily story view
  - Rating interface (1-5 stars)
  - Story history
- [ ] Integrate with FastAPI backend
- [ ] Test with real users (goal: 50+ rated stories per group)

### Phase 4: Preference Learning (v2) - After 1-2 Months
- [ ] Analyze collected `prompt_performance` data
- [ ] Implement preference embedding calculation:
  - Average embeddings of highly-rated stories per group
  - Extract theme/type preferences from ratings
  - Populate `group_preferences` table
- [ ] Update prompt builder to use weighted random selection:
  - 80% exploit (use learned preferences)
  - 20% explore (pure random)
- [ ] A/B test v1 (random) vs v2 (learned) to validate improvement
- [ ] Deploy v2 if ratings improve

### Phase 5: Production & Launch
- [ ] Polish iOS app UI/UX
- [ ] Add push notifications for daily stories
- [ ] Performance optimization
- [ ] App Store submission
- [ ] Marketing & user acquisition

## Key Decisions Summary

### v1 vs v2 Approach

| Aspect | v1 (MVP) | v2 (After Data Collection) |
|--------|----------|---------------------------|
| **Theme Selection** | Pure random from list | Weighted random based on group's past ratings |
| **Story Type Selection** | Pure random | Weighted random based on what group rates highly |
| **Character Selection** | Random 2-3 from group | Still random (or could weight by who appears in high-rated stories) |
| **Personalization** | 20% of days, random detail | 20% of days, semantically matched detail to theme |
| **Goal** | Collect ratings data | Use ratings data to improve |
| **Database** | Stores everything, analyzes nothing | Analyzes `prompt_performance` to build `group_preferences` |
| **Expected Rating** | 3.5+/5 | 4.0+/5 (validated via A/B test) |

### Why Start with v1?
1. **Faster to market** - Simpler logic, fewer edge cases
2. **Real data** - Need actual user ratings to know if v2 will help
3. **Avoid premature optimization** - Might discover random works fine!
4. **Database ready** - Schema supports both, easy upgrade path

### Tech Stack Summary

**Training:** PyTorch, HuggingFace Transformers, PEFT (LoRA)
**Model:** Llama-3.1-8B-Instruct fine-tuned on 106K absurdist story fragments
**Backend:** FastAPI (Python)
**Database:** Supabase (PostgreSQL + pgvector for embeddings)
**Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Frontend:** iOS app (Swift/SwiftUI)
**Deployment:** Backend on Railway/Render, Model on GPU instance (RunPod/Lambda)

## License

MIT License
