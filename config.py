"""
Configuration file for Fiction Unlimited project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEMPLATES_DIR = DATA_DIR / "templates"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TEMPLATES_DIR, MODELS_DIR, DATABASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Data collection settings
# TODO: edit these 
TARGET_STORIES_COUNT = 10000
MIN_STORY_LENGTH = 10  # words
MAX_STORY_LENGTH = 50  # words
IDEAL_STORY_LENGTH_MIN = 10
IDEAL_STORY_LENGTH_MAX = 50

# AO3 scraping settings
# TODO: add more tags 
AO3_TAGS = [
    "Humor",
    "Crack",
    "Plot Twists",
    "Absurdism",
    "Ambiguity",
    "Awkwardness",
    "Badass", 
    "Bathing/Washing", 
    "BDSM", 
    "Biting",
    "Bottoming",
    "Boys In Love", 
    "Confessions", 
    "Crushes",
    "Cuddling & Snuggling",
    "Dark",
    "Drama",
    "Drugs",
    "Enemies",
    "Falling In Love", 
    "Feelings",
    "Feels",
    "Female Relationships",
    "Fights",
    "Flirting", 
    "Fluff",
    "Food",
    "Getting Together", 
    "Idiots in Love",
    "Intoxication",
    "Jealousy",
    "Kinks",
    "Kissing",
    "LGBTQ Themes",
    "Light-Hearted",
    "Love",
    "Making Out",
    "Misunderstandings",
    "Praise Kink",
    "Relationship(s)",
    "Romance",
    "Science Fiction & Fantasy",
    "Short",
    "Smut",
    "Teasing",
    "Topping",
    "Satire",
    "Comedy",
    "Ridiculous",
    "Chaos",
    "Funny",
]
# TODO: increase number of stories to scrape 
AO3_MAX_WORKS = 5000

# Model settings
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training settings
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
WARMUP_STEPS = 100

# Generation settings
GENERATION_TEMPERATURE = 0.9
GENERATION_TOP_P = 0.95
GENERATION_TOP_K = 50
MAX_NEW_TOKENS = 200
MIN_STORY_TOKENS = 50

# Embedding settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ChromaDB settings
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_data")
SCENARIO_COLLECTION = "scenarios"
PLOT_TWIST_COLLECTION = "plot_twists"
ARCHETYPE_COLLECTION = "archetypes"
USER_HISTORY_COLLECTION = "user_history"

# Template targets
NUM_SCENARIOS = 500
NUM_PLOT_TWISTS = 300
NUM_ARCHETYPES = 100

# Recency filtering
SCENARIO_RECENCY_DAYS = 30
PLOT_TWIST_RECENCY_DAYS = 14

# Retrieval settings
PERSONALITY_WEIGHT = 0.6
PREFERENCE_WEIGHT = 0.4
TOP_K_SCENARIOS = 5
TOP_K_PLOT_TWISTS = 3

# Evaluation targets
TARGET_PERPLEXITY = 50.0
TARGET_SELF_BLEU = 0.3
TARGET_USER_RATING = 3.5
TARGET_SEMANTIC_DISSIMILARITY = 0.7

# Database settings
DATABASE_PATH = DATABASE_DIR / "fiction_unlimited.db"

# Feedback settings
HIGH_RATING_THRESHOLD = 4  # Ratings >= 4 are considered "high"
PREFERENCE_UPDATE_WEIGHT = 0.3  # How much new ratings influence preferences

# API settings (for production)
API_HOST = "0.0.0.0"
API_PORT = 8000
