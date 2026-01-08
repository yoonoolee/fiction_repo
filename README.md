# FriendFic - AI Story Generation System

A personalized daily story generation system that creates absurdist, humorous narratives for friend groups using fine-tuned LLMs, preference learning, and triple hybrid RAG (fine-tuned model + training data examples + group-specific favorites).

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
2. **Prompt Builder**:
   - v1: Random selection of story type, characters, theme, and optional personalization
   - v2: Weighted random selection using preference embeddings + triple hybrid RAG
3. **User Data Store**: Database storing friend attributes (pets, interests, siblings, partners) and generated stories
4. **Theme Library**: Curated list of ~15-20 themes/emotions/genres
5. **Triple Hybrid RAG (v2)**: Fine-tuned model + training data examples + group-specific favorites
6. **Preference Learning (v2)**: Per-group embeddings for theme/type weighting based on ratings
7. **Evaluation Framework**: Perplexity, Self-BLEU, diversity metrics, and user ratings

### Tech Stack

- **ML Framework**: PyTorch, HuggingFace Transformers, PEFT
- **Database**: PostgreSQL + pgvector (via Supabase) - stores friend data, training stories, generated stories, embeddings
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Backend**: FastAPI (Python)
- **RAG (v2)**: Training data retrieval + per-group favorites retrieval (triple hybrid)
- **Model Deployment**: HuggingFace Spaces (free tier, CPU with 4-bit quantization)
- **Generation Strategy**: Scheduled batch generation (pre-generated stories cached in database)
- **Frontend**: iOS app (Swift/SwiftUI)
- **Cost**: ~$0-5/month (HuggingFace Spaces free, Supabase free tier, optional App Store $99/year)

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

**v2 (Future - After 1-2 months):** Preference learning + Triple Hybrid RAG
- Analyze collected rating data
- Build preference embeddings per group (learned "taste profile")
- Switch to weighted random selection based on similarity to theme/type embeddings
- **Training data RAG:** Retrieve theme-matched examples from 106K training stories (general style)
- **Group favorites RAG:** Retrieve each group's highly-rated stories as examples (group-specific)
- System improves over time automatically as more ratings collected

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

### Daily Story Pipeline (v2 - Triple Hybrid RAG)

1. **Select Story Type** (weighted random)
   - Calculate similarity between group's preference embedding and each story type embedding
   - Convert similarities to weights (80% exploit preferred types, 20% explore random)
   - Example: Group prefers one-liners → higher weight for one-liners

2. **Select Characters** (random)
   - Pick 2-3 friends from the friend group

3. **Select Theme/Emotion/Genre** (weighted random)
   - Calculate similarity between group's preference embedding and each theme embedding
   - Convert similarities to weights
   - Example: If group's preference embedding is similar to "chaos" embedding (0.85) vs "romance" (0.41) → chaos gets higher weight

4. **RAG #1: Retrieve Training Data Examples** (NEW!)
   - Query original 106K training dataset for stories matching selected theme
   - Filter: Stories tagged/matching the chosen theme (e.g., "chaos")
   - Retrieve 1-2 random examples
   - Purpose: Ground in general absurdist style for this theme

5. **RAG #2: Retrieve Group's Favorite Stories** (NEW!)
   - Query vector DB for similar stories matching the selected theme
   - Filter: `rating >= 4` AND `group_id = current_group` AND `theme = selected_theme`
   - Retrieve 2-3 examples using cosine similarity
   - Purpose: Match this specific group's humor preferences

6. **Personalization** (20% of days, optional)
   - Random selection: include one friend's attribute (pet, interest, sibling, partner)
   - Future: Could use RAG to semantically match attributes to theme

7. **Generate Story with Triple Hybrid RAG**
   - Build prompt with: training examples + group's favorite examples + theme + characters + optional personalization
   - Prompt structure:
     - "Style references: [training data examples]"
     - "This group loved: [group's rated stories]"
     - "Create new [theme] story with [characters]"
   - Send to fine-tuned Llama-3.1-8B-Instruct
   - Return story

### Example Prompts

**v1 - Without Personalization:**
```
Create a one-liner (10-20 words) featuring Alex and Padma in a romance story.
```

**v1 - With Personalization:**
```
Create a short story (20-40 words) featuring Manasa and Sam in a chaos story.
Include that Manasa has a cat named Remy.
```

**v2 - With Triple Hybrid RAG (Training Data + Group Favorites):**
```
Style references from training data:
- "I accidentally joined a pyramid scheme for essential oils. Now I'm essential oils."
- "Told my cat I loved them. Cat filed a restraining order."

This friend group loved these stories:
- "Alex and Padma accidentally started a cult while waiting for pizza." (5★)
- "Sam declared war on the local geese. The geese won." (4★)

Create a new chaos short story (20-40 words) featuring Manasa and Sam in a similar style.
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

-- Training Data (106K stories for RAG retrieval)
CREATE TABLE training_stories (
    story_id UUID PRIMARY KEY,
    story_text TEXT,
    story_embedding VECTOR(384),

    -- Metadata for filtering
    source TEXT,                  -- "reddit_tifu", "reddit_confessions", "ao3"
    theme_tags JSONB,             -- ["chaos", "absurdist", "comedy"]
    word_count INTEGER,

    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_embedding USING ivfflat (story_embedding vector_cosine_ops)
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
│  - Serve cached stories │
│  - Store ratings        │
│  - Trigger generation   │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐    ┌──────────────────────┐
│  Supabase                │    │  HuggingFace Spaces  │
│  (PostgreSQL + pgvector) │◄───│  (Free Tier)         │
│  - Friend data           │    │  - Fine-tuned model  │
│  - Training stories (RAG)│    │  - 4-bit quantized   │
│  - Generated stories     │    │  - CPU inference     │
│  - Story embeddings      │    │  - Batch generation  │
│  - Ratings & prompts     │    └──────────────────────┘
│  - Preference embeddings │
└──────────────────────────┘

Flow: Scheduled job → HuggingFace API → Generate stories → Cache in Supabase → iOS fetches
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
- After collecting 50+ ratings, calculate average embedding of highly-rated stories per group
- This creates a "taste profile" vector for each group (one embedding = 384 numbers)
- Use it to weight random selection toward themes/types this group prefers

**How theme weighting works:**
1. Each theme ("chaos", "romance", etc.) has its own embedding
2. Calculate similarity between group's preference embedding and each theme embedding
3. Convert similarities to percentages (weights)
4. Example: Group's preference embedding similar to "chaos" (0.85) → chaos gets 30% weight vs "romance" (0.41) → romance gets 14% weight

**v1 (Now):** Just collect the data
- Store all prompts + ratings in `prompt_performance` table
- Don't act on it yet

**v2 (Later):** Analyze and apply
- Build `group_preferences` from `prompt_performance` data (one preference embedding per group)
- Switch from pure random to weighted random selection
- 80% exploit (use learned preferences), 20% explore (random)

### 3. Triple Hybrid RAG: Training Data + Group Favorites (v2)
**Problem:** How to ensure new stories match both the general absurdist style AND each group's specific humor preferences?

**Solution:** Use dual RAG retrieval - general style examples from training data + group-specific examples from rated stories

**Implementation - RAG #1: Training Data Examples**
```sql
-- Query training dataset for general style examples matching theme
SELECT story_id, story_text
FROM training_stories
WHERE theme_tags @> '["chaos"]'::jsonb  -- Match selected theme
ORDER BY RANDOM()
LIMIT 2;
```

**Implementation - RAG #2: Group's Favorite Stories**
```sql
-- Query group's highly-rated stories matching theme
SELECT story_id, story_text, rating,
       1 - (story_embedding <=> query_embedding) AS similarity
FROM generated_stories
WHERE group_id = $1
  AND rating >= 4
  AND theme = $2  -- Match selected theme
  AND created_at > NOW() - INTERVAL '60 days'
ORDER BY story_embedding <=> query_embedding
LIMIT 3;
```

**How it works:**
1. Select theme using weighted random (preference embedding) → e.g., "chaos"
2. **RAG #1:** Query training dataset for 1-2 "chaos" examples → general absurdist style
3. **RAG #2:** Query this group's highly-rated "chaos" stories → group-specific humor
4. Include both in prompt:
   - "Style references: [training examples]"
   - "This group loved: [group's rated stories]"
   - "Create new chaos story..."
5. Model learns from both general style AND group-specific preferences

**Benefits:**
- **Training data RAG:** Grounds in general absurdist style for the theme
- **Group favorites RAG:** Matches specific group's humor/tone preferences
- **Triple hybrid:** Fine-tuned model + training examples + group examples = highly personalized output

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
- [ ] Create database schema (all 6 tables including `training_stories` and v2-ready `group_preferences`)
- [ ] **Load training data into database:**
  - Process 106K training stories
  - Generate embeddings for each story
  - Tag stories with themes (chaos, romance, etc.)
  - Populate `training_stories` table
- [ ] Create theme/emotion/genre library (~15-20 themes)
- [ ] Build FastAPI backend:
  - Random prompt builder (type, theme, characters, optional personalization)
  - Story deduplication with vector similarity
  - Store generated stories + embeddings
  - Store prompt performance data (for future v2)
  - Rating endpoint (updates both `generated_stories` and `prompt_performance`)
- [ ] Deploy fine-tuned model to HuggingFace Spaces (free tier, 4-bit quantization for CPU)
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

### Phase 4: Preference Learning + Triple Hybrid RAG (v2) - After 1-2 Months
- [ ] Analyze collected `prompt_performance` data
- [ ] Implement preference embedding calculation:
  - Average embeddings of highly-rated stories per group (one embedding per group)
  - Calculate similarity between group preference and each theme/type embedding
  - Populate `group_preferences` table
- [ ] Update prompt builder to use weighted random selection:
  - Calculate weights from preference similarities
  - 80% exploit (use learned preferences), 20% explore (pure random)
- [ ] **Implement triple hybrid RAG system:**
  - **RAG #1:** Query `training_stories` for 1-2 theme-matched examples (general style)
  - **RAG #2:** Query `generated_stories` for 2-3 highly-rated group stories (rating >= 4, group-specific)
  - Build prompts with both: "Style references: [training]... This group loved: [group stories]..."
  - Inject both sets of examples into prompt
  - Optional: RAG for semantic attribute matching (personalization)
- [ ] Test v2 with real users to validate improvement over v1
- [ ] Deploy v2 if ratings improve

### Phase 5: Production & Launch
- [ ] Polish iOS app UI/UX
- [ ] Deploy model to HuggingFace Spaces (4-bit quantization for CPU inference)
- [ ] Set up scheduled batch generation (cron job to generate stories and cache in database)
- [ ] Add push notifications for daily stories
- [ ] Performance optimization
- [ ] TestFlight testing (free) or App Store submission ($99/year)
- [ ] Marketing & user acquisition

### Deployment Strategy (Low-Cost/Free)
- **Model:** HuggingFace Spaces free tier (16GB RAM, CPU, 48hr sleep timeout)
- **Backend:** FastAPI + scheduled job to hit HuggingFace API daily
- **Database:** Supabase free tier (500MB storage)
- **Generation:** Batch pre-generation (10-30 sec per story via CPU is fine for overnight jobs)
- **Caching:** Generated stories cached in Supabase, served instantly to iOS app
- **Total Cost:** ~$0/month (or $99/year if deploying to App Store instead of TestFlight)

## Key Decisions Summary

### v1 vs v2 Approach

| Aspect | v1 (MVP) | v2 (After Data Collection) |
|--------|----------|---------------------------|
| **Theme Selection** | Pure random from list | Weighted random via similarity: preference embedding vs theme embeddings |
| **Story Type Selection** | Pure random | Weighted random via similarity: preference embedding vs type embeddings |
| **Character Selection** | Random 2-3 from group | Still random |
| **Personalization** | 20% of days, random detail | 20% of days, random detail (future: RAG-matched to theme) |
| **RAG (Retrieval)** | ❌ None | ✅ **Triple hybrid:** Fine-tuned model + training data RAG + group favorites RAG |
| **Training Data RAG** | ❌ None | ✅ Retrieve 1-2 theme-matched examples from 106K training stories (general style) |
| **Group Favorites RAG** | ❌ None | ✅ Retrieve 2-3 highly-rated stories from this group's history (group-specific) |
| **Preference Embedding** | ❌ None | ✅ One embedding per group (average of their highly-rated stories) |
| **Prompt Style** | Simple instruction | "Style refs: [training]... Group loved: [group stories]... Create new story" |
| **Goal** | Collect ratings data | Use ratings for weighted selection + dual RAG retrieval |
| **Expected Rating** | 3.5+/5 | 4.0+/5 (validated via user testing) |

### v2 Triple Hybrid Approach: Fine-Tuned Model + Training Data RAG + Group Favorites RAG

**Three layers working together:**

1. **Fine-Tuned Model** (base capability)
   - Llama-3.1-8B trained on 106K absurdist stories
   - Knows general absurdist narrative style
   - Provides foundational story generation ability

2. **Preference Embedding** (theme/type weighting)
   - One embedding per group (average of their highly-rated stories)
   - Calculate similarity to each theme/type embedding
   - Influences *what* to generate (which themes/types to favor)

3. **Training Data RAG** (general style examples)
   - Query 106K training stories for theme-matched examples
   - Retrieve 1-2 stories tagged with selected theme
   - Grounds generation in general absurdist style for that specific theme

4. **Group Favorites RAG** (group-specific style examples)
   - Query this group's highly-rated stories (rating >= 4)
   - Retrieve 2-3 stories matching selected theme
   - Influences *how* to generate (this group's specific humor/tone preferences)

**Complete Example - Group "The Roommates":**

1. **Preference embedding** → Theme similarity: chaos (0.85) vs romance (0.41) → Select "chaos" (weighted random)
2. **Training data RAG** → Retrieve: "I accidentally joined a pyramid scheme for essential oils. Now I'm essential oils."
3. **Group favorites RAG** → Retrieve: "The roommates declared war on a rogue Roomba. The Roomba won." (5★, from this group's history)
4. **Final prompt:** "Style reference: [training example]... This group loved: [group's chaos story]... Create new chaos story with Alex and Sam"
5. **Result:** Story matches general absurdist chaos style AND The Roommates' specific humor preferences

### Why Start with v1?
1. **Faster to market** - Simpler logic, fewer edge cases
2. **Real data** - Need actual user ratings to know if v2/RAG will help
3. **Avoid premature optimization** - Might discover random works fine!
4. **Database ready** - Schema supports both, easy upgrade path

### Tech Stack Summary

**Training:** PyTorch, HuggingFace Transformers, PEFT (LoRA)
**Model:** Llama-3.1-8B-Instruct fine-tuned on 106K absurdist story fragments
**Backend:** FastAPI (Python)
**Database:** Supabase (PostgreSQL + pgvector for embeddings)
**Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**RAG (v2):**
- Training data RAG: Query 106K training stories by theme tags
- Group favorites RAG: Vector similarity search on per-group highly-rated stories
- Both injected as examples in prompts (triple hybrid approach)
**Model Deployment:** HuggingFace Spaces (free tier, CPU with 4-bit quantization)
**Generation:** Scheduled batch generation with database caching
**Frontend:** iOS app (Swift/SwiftUI)
**Cost:** ~$0-5/month (free tiers for all services)

## License

MIT License
