"""
Generate absurd stories using Groq API (free tier)

No training required - uses Llama-3.1-8B-Instruct with few-shot prompting
"""

import os
import random
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Initialize Groq client
# Get free API key from: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Get one from https://console.groq.com/keys")

client = Groq(api_key=GROQ_API_KEY)

# Load training data for few-shot examples
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "test" / "tifu_ao3.csv"
print(f"Loading examples from: {DATA_PATH}")
story_df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(story_df)} example stories")

def generate_absurd_story(story_type, characters, num_examples=1, print_examples=False):
    """
    Generate an absurd story

    Args:
        story_type: "short story" (20-40 words) or "one liner" (10-20 words)
        characters: List of character names (e.g., ["Alex", "Sam"])
        num_examples: Number of example stories to include in prompt
        print_examples: If True, print the sampled examples

    Returns:
        Generated story string
    """

    # Filter by story type based on source
    if story_type.lower() == "one liner":
        # Get TIFU stories (one-liners)
        filtered_df = story_df[story_df['id'].str.startswith('tifu_')]
    else:
        # Get AO3 stories (short stories)
        filtered_df = story_df[story_df['id'].str.startswith('ao3_')]

    # Sample random examples from filtered data
    sample_stories = filtered_df.sample(n=num_examples)

    # Print examples if requested
    if print_examples:
        print(f"\nRandomly selected {num_examples} few-shot examples:")
        print("-" * 80)
        for i, (_, row) in enumerate(sample_stories.iterrows(), 1):
            story_text = str(row['text']).strip()
            if story_text.startswith('"') and story_text.endswith('"'):
                story_text = story_text[1:-1].strip()
            print(f"{i}. {story_text}")
        print("-" * 80 + "\n")

    # Build examples string
    examples = "Here are examples of the style of absurd stories to generate:\n\n"

    for i, (_, row) in enumerate(sample_stories.iterrows(), 1):
        story_text = str(row['text']).strip()
        # Clean up the text
        if story_text.startswith('"') and story_text.endswith('"'):
            story_text = story_text[1:-1].strip()

        examples += f"Example {i}:\n{story_text}\n\n"

    # Format character list
    char_str = ", ".join(characters) if isinstance(characters, list) else characters

    # Adjust instructions based on story type
    if story_type.lower() == "one liner":
        length_instruction = "Create a brief one-liner story (10-20 words) with the same content as the example given."
    else:
        length_instruction = "Create a short absurd story (20-40 words) with the same content as the example given."

    # Build prompt
    prompt = f"""{examples}

{length_instruction} featuring the characters: {char_str}

Story:"""

    # Generate with Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a creative writer specializing in brief stories. The content should be similar to the example. Keep stories between 10-40 words. Never be depressing or serious."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9,
        max_tokens=100,
        top_p=0.95,
    )

    story = response.choices[0].message.content.strip()

    # Remove "Story: " prefix if model included it
    if story.startswith("Story:"):
        story = story[6:].strip()

    return story


# ============================================================================
# Test the generator
# ============================================================================

if __name__ == "__main__":
    print("Absurd Story Generator")
    print("=" * 80)
    print("Type 'quit' to exit\n")

    while True:
        story_type = input("\nStory type (short story / one liner): ")
        if story_type.lower() == 'quit':
            break

        char_input = input("Enter characters (comma-separated): ")
        characters = [c.strip() for c in char_input.split(",")]

        # Generate with examples printed
        story = generate_absurd_story(story_type, characters, print_examples=True)

        print(f"📖 Generated Story: {story}")
        print(f"📊 Word count: {len(story.split())}")
        print("=" * 80)
