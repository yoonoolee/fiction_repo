# Fiction Unlimited - AI Story Generation System

A personalized daily story generation system that creates absurdist, humorous narratives for friend groups using fine-tuned LLMs, RAG-based personalization, and user feedback loops.

## Project Overview

This system generates unique, personalized short stories for friend groups every day. Stories are:
- Absurdist and humorous with unexpected plot twists
- Personalized based on friend group personalities and preferences
- 50-200 words in length
- Never repetitive (30-day scenario recency filter, 14-day twist filter)

## Architecture

### ML Components

1. **Fine-tuned Story Generator**: Llama-3.1-8B with LoRA adapters trained on absurdist stories
2. **Personalization System**: Sentence-transformer embeddings for matching personalities to story elements
3. **RAG System**: ChromaDB vector database with 800+ templates (scenarios, plot twists, archetypes)
4. **Feedback Loop**: Preference learning from user ratings (1-5 stars)
5. **Evaluation Framework**: Perplexity, Self-BLEU, and semantic similarity metrics

### Tech Stack

- **ML Framework**: PyTorch, HuggingFace Transformers, PEFT
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (local), Pinecone (production)
- **Database**: SQLite (dev), PostgreSQL (production)
- **Backend**: FastAPI
- **Demo UI**: Streamlit
- **Deployment**: iOS app (Swift/SwiftUI)

## Project Structure

```
fiction-unlimited/
 data/
    raw/              # Scraped training data
    processed/        # Cleaned and split datasets
    templates/        # Scenario, plot twist, archetype templates
 notebooks/            # Jupyter notebooks for experiments
 src/
    scraping/         # Web scraping scripts
    preprocessing/    # Data cleaning and preparation
    training/         # Model fine-tuning scripts
    rag/             # RAG system and embeddings
    generation/      # Story generation pipeline
    evaluation/      # Metrics and evaluation
 models/              # Saved model weights and LoRA adapters
 database/            # SQLite database files
 streamlit/           # Demo interface
 requirements.txt     # Python dependencies
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
python src/training/fine_tune.py
```

### 5. Set Up RAG System

```bash
python src/rag/setup_chromadb.py
```

### 6. Run Demo

```bash
streamlit run streamlit/app.py
```

## Performance Targets

- **Perplexity**: < 50 (coherent, natural language)
- **Self-BLEU**: < 0.3 (diverse outputs)
- **User Rating**: > 3.5/5 average
- **Generation Latency**: < 5 seconds
- **No repetition**: 30 days (scenarios), 14 days (twists)

## Development Roadmap

### Phase 1: Data & Training
- [x] Set up project structure
- [ ] Scrape 10k+ training stories
- [ ] Clean and preprocess data
- [ ] Create template libraries
- [ ] Fine-tune Llama-3.1-8B with LoRA

### Phase 2: RAG & Pipeline
- [ ] Set up ChromaDB
- [ ] Generate template embeddings
- [ ] Build retrieval system
- [ ] Implement generation pipeline
- [ ] Add recency filtering

### Phase 3: Evaluation & Feedback
- [ ] Implement evaluation metrics
- [ ] Build feedback loop
- [ ] Test with real users
- [ ] Iterate on quality

### Phase 4: Demo & iOS App
- [ ] Create Streamlit demo
- [ ] Build iOS app with SwiftUI
- [ ] Integrate backend API
- [ ] Deploy to App Store

## License

MIT License
