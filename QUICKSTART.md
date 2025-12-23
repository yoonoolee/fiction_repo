# Quick Start Guide

## Phase 1: Data Collection (Do This First!)

### Step 1: Install dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download Reddit Stories (No API Needed!)

Use pre-collected datasets from HuggingFace:

```bash
# Option 1: Run the download script
python src/scraping/download_reddit_datasets.py

# Option 2: Use the interactive Jupyter notebook (recommended)
jupyter notebook notebooks/download_reddit_datasets.ipynb
```

This downloads 66k+ Reddit WritingPrompts stories instantly - no API credentials needed!

### Step 3: Set up API credentials (Optional)

Only needed for HuggingFace model downloads later:

```bash
cp .env.template .env
# Edit .env and add your HuggingFace token
```

**Get HuggingFace token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token
3. Add to .env

### Step 4: Clean and preprocess data

```bash
# Clean the scraped data and split into train/val/test
python src/preprocessing/clean_data.py
```

### Step 5: Create template libraries

```bash
# Generate starter templates
python src/preprocessing/create_templates.py

# Then expand templates to reach target numbers:
# - Edit data/templates/scenarios.txt (target: 500)
# - Edit data/templates/plot_twists.txt (target: 300)
# - Edit data/templates/archetypes.txt (target: 100)
```

**Tip:** Use Claude or GPT to generate more templates quickly!

---

## Phase 2: Model Fine-tuning

### Option A: Google Colab (Recommended for beginners)

1. Open `notebooks/fine_tune_llama.ipynb` in Colab
2. Connect to GPU runtime (Runtime > Change runtime type > T4 GPU)
3. Run cells to fine-tune
4. Download LoRA weights when done

**Cost:** $10/month for Colab Pro (worth it for better GPUs)

### Option B: Local GPU

If you have a GPU with 16GB+ VRAM:

```bash
# Run training script
python src/training/fine_tune.py
```

---

## Phase 3: RAG System Setup

```bash
# Set up ChromaDB and embed templates
python src/rag/setup_chromadb.py
```

---

## Phase 4: Database Setup

```bash
# Create SQLite database
python src/database/create_db.py

# Add sample friend groups
python src/database/create_samples.py
```

---

## Phase 5: Test Generation

```bash
# Generate test stories
python src/generation/generate_story.py
```

---

## Phase 6: Evaluation

```bash
# Run evaluation metrics
python src/evaluation/evaluate.py
```

---

## Phase 7: Demo

```bash
# Run Streamlit demo
streamlit run streamlit/app.py
```

---

## Timeline Estimate

| Phase | Time Estimate | Can Run Overnight? |
|-------|--------------|-------------------|
| Data collection | 6-8 hours | Yes (AO3 scraping) |
| Data preprocessing | 10 minutes | No |
| Template creation | 2-4 hours | No (manual work) |
| Model fine-tuning | 2-6 hours | Yes (depends on GPU) |
| RAG setup | 30 minutes | No |
| Database setup | 15 minutes | No |
| Generation pipeline | 2 hours | No |
| Evaluation | 1 hour | No |
| Streamlit demo | 3 hours | No |

**Total:** ~20-30 hours of active work + overnight runs

---

## Troubleshooting

**"Out of memory" during fine-tuning**
- Reduce batch size in config.py
- Use gradient checkpointing
- Use Google Colab with GPU instead

**AO3 scraping too slow**
- This is expected (respecting rate limits)
- Run overnight or skip AO3 entirely
- Reddit alone provides plenty of data

---

## Next: iOS App Development

Once the ML system works, we'll build the iOS app:
1. Learn Swift/SwiftUI basics
2. Set up Xcode project
3. Build UI for friend groups and story display
4. Connect to backend API
5. Deploy to App Store

This will take another 2-4 weeks depending on iOS experience.
