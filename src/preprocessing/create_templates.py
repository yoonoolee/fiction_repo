"""
Template library creation script
Helps create scenario, plot twist, and character archetype templates
Can start with pre-made examples and expand manually
"""
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TEMPLATES_DIR, NUM_SCENARIOS, NUM_PLOT_TWISTS, NUM_ARCHETYPES


# Starter templates - you'll expand these to reach target numbers
STARTER_SCENARIOS = [
    "art thieves in Paris",
    "rival food truck owners",
    "accidental time travelers",
    "competitive dog groomers",
    "ghost hunters at a furniture store",
    "astronauts on a failed mission",
    "wedding planners with clashing visions",
    "treasure hunters in a suburban basement",
    "rival librarians",
    "accidentally swapped bodies",
    "stuck in an escape room",
    "competing for the same apartment",
    "amateur detectives solving a mystery",
    "lost in IKEA",
    "accidentally joined a cult",
    "neighbors in a prank war",
    "competitive bakers",
    "stranded on a deserted island",
    "accidentally became viral internet celebrities",
    "roommates with very different lifestyles",
    "trapped in an elevator",
    "competing on a reality show",
    "accidentally inherited a weird business",
    "medieval reenactors taking it too seriously",
    "accidentally discovered a portal to another dimension",
    "competing chefs at a food festival",
    "accidentally adopted the same dog",
    "stuck in a time loop",
    "accidentally became superheroes",
    "rival tour guides",
    "accidentally started a revolution",
    "competing for the same job",
    "accidentally robbed a bank",
    "stuck on a broken ski lift",
    "accidentally summoned a demon",
    "rival antique dealers",
    "accidentally became spies",
    "competing in an underground fight club",
    "accidentally inherited a haunted house",
    "stuck at a terrible party",
]

STARTER_PLOT_TWISTS = [
    "they liked it",
    "the cat was the mastermind",
    "it was all a dream (but not really)",
    "they were the same person",
    "everyone was a robot except them",
    "it happened on Mars",
    "they're actually in the past",
    "the conflict resolved itself off-screen",
    "they fell in love",
    "they were dead the whole time",
    "it was a misunderstanding about sandwiches",
    "the villain won but nobody noticed",
    "it was all an elaborate joke",
    "they became best friends instead",
    "the real treasure was the friends they made",
    "it was filmed for a reality show",
    "they woke up and it was five years later",
    "aliens were involved",
    "it was a test and they failed",
    "they realized they were in a simulation",
    "the dog could talk and had been judging them",
    "they were immortal and forgot",
    "it was their birthday the whole time",
    "they accidentally saved the world",
    "they accidentally destroyed the world (but it's fine)",
    "everyone was actually their relative",
    "they were in a different country than they thought",
    "it was actually a musical",
    "they switched to the wrong timeline",
    "the narrator was lying",
]

STARTER_ARCHETYPES = [
    "chaos gremlin - causes problems on purpose",
    "tired mom friend - keeps everyone alive",
    "chaotic neutral - only cares about fun",
    "anxious overthinker - worst case scenario expert",
    "reckless optimist - everything will work out (it won't)",
    "sarcastic pessimist - expects the worst, still disappointed",
    "impulsive disaster - acts before thinking",
    "paranoid conspiracy theorist - always right but for wrong reasons",
    "overly competitive - must win at everything",
    "hopeless romantic - sees love everywhere",
    "practical engineer - has a tool for everything",
    "dramatic theater kid - everything is a performance",
    "cryptid enthusiast - knows too much about weird stuff",
    "caffeine-dependent insomniac - hasn't slept in days",
    "plant parent - cares more about plants than people",
    "thrill seeker - adrenaline junkie",
    "conspiracy theory debunker - trust no one, verify everything",
    "chronic people pleaser - can't say no",
    "rebellious rule breaker - authority? never heard of her",
    "nostalgic overthinker - living in the past",
]


def save_templates_to_file(templates, filename, description):
    """Save templates to a text file (one per line)"""
    output_path = TEMPLATES_DIR / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        for template in templates:
            f.write(template + '\n')

    print(f" Saved {len(templates)} {description} to {output_path}")

    return output_path


def save_templates_to_json(templates, filename, template_type):
    """Save templates to JSON format with metadata"""
    output_path = TEMPLATES_DIR / filename

    template_list = []
    for i, template in enumerate(templates, 1):
        template_list.append({
            "id": f"{template_type}_{i}",
            "text": template,
            "type": template_type,
            "usage_count": 0,
            "average_rating": None
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_list, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(templates)} {template_type} to {output_path} (JSON)")

    return output_path


def generate_more_scenarios():
    """Generate variations of existing scenarios"""
    # This is where you could add more scenarios
    # For now, we'll just add some variations
    variations = []

    locations = [
        "in a submarine", "at a retirement home", "in outer space",
        "at a renaissance fair", "in a shopping mall", "at a spa",
        "in a library", "at a theme park", "on a cruise ship"
    ]

    occupations = [
        "competitive knitters", "rival podcasters", "amateur magicians",
        "conspiracy theorist bloggers", "pet psychics", "professional cuddlers",
        "escape room designers", "food critics", "social media influencers"
    ]

    situations = [
        "accidentally started a business",
        "competing in a bizarre competition",
        "stuck in an awkward situation",
        "accidentally became famous",
        "trying to break a world record"
    ]

    for occupation in occupations:
        variations.append(occupation)

    for situation in situations:
        variations.append(situation)

    return variations


def generate_more_plot_twists():
    """Generate more plot twist variations"""
    variations = []

    # Add more absurdist twists
    twists = [
        "it was the wrong [person/place]",
        "they were in a [thing] the whole time",
        "the real [thing] was [unexpected thing]",
        "they learned nothing",
        "it got worse somehow",
        "they succeeded but at what cost (nothing actually)",
        "the problem solved itself via coincidence",
        "they discovered they had [weird power]",
        "time travel was involved (poorly)",
        "it was all in the terms and conditions",
    ]

    variations.extend(twists)

    return variations


def print_expansion_guide():
    """Print guide for expanding templates"""
    print("\n" + "=" * 60)
    print(" Template Expansion Guide")
    print("=" * 60)
    print(f"\nCurrent template counts:")
    print(f"  Scenarios: {len(STARTER_SCENARIOS)} / {NUM_SCENARIOS} (target)")
    print(f"  Plot Twists: {len(STARTER_PLOT_TWISTS)} / {NUM_PLOT_TWISTS} (target)")
    print(f"  Archetypes: {len(STARTER_ARCHETYPES)} / {NUM_ARCHETYPES} (target)")

    print(f"\n To reach target numbers, you can:")
    print(f"  1. Edit the STARTER_* lists in this file")
    print(f"  2. Generate variations programmatically")
    print(f"  3. Use an LLM to generate more (recommended)")
    print(f"  4. Manually write more in the .txt files")

    print(f"\n Recommended: Use Claude or GPT to generate variations")
    print(f"  Example prompt:")
    print(f'  "Generate 50 absurdist scenario ideas similar to:')
    print(f'   - art thieves in Paris')
    print(f'   - rival food truck owners')
    print(f'   - accidental time travelers"')


def main():
    """Main template creation function"""
    print("=" * 60)
    print("Template Library Creation")
    print("=" * 60)

    # Expand templates with variations
    all_scenarios = STARTER_SCENARIOS + generate_more_scenarios()
    all_plot_twists = STARTER_PLOT_TWISTS + generate_more_plot_twists()
    all_archetypes = STARTER_ARCHETYPES

    # Remove duplicates
    all_scenarios = list(set(all_scenarios))
    all_plot_twists = list(set(all_plot_twists))
    all_archetypes = list(set(all_archetypes))

    print(f"\n Created template libraries:")
    print(f"  Scenarios: {len(all_scenarios)}")
    print(f"  Plot Twists: {len(all_plot_twists)}")
    print(f"  Archetypes: {len(all_archetypes)}")

    # Save as text files (easy to edit manually)
    print(f"\n Saving templates (text format)...")
    save_templates_to_file(all_scenarios, "scenarios.txt", "scenarios")
    save_templates_to_file(all_plot_twists, "plot_twists.txt", "plot twists")
    save_templates_to_file(all_archetypes, "archetypes.txt", "archetypes")

    # Also save as JSON (for programmatic use)
    print(f"\n Saving templates (JSON format)...")
    save_templates_to_json(all_scenarios, "scenarios.json", "scenario")
    save_templates_to_json(all_plot_twists, "plot_twists.json", "plot_twist")
    save_templates_to_json(all_archetypes, "archetypes.json", "archetype")

    # Print expansion guide
    print_expansion_guide()

    # Print sample templates
    print(f"\n" + "=" * 60)
    print(" Sample Templates")
    print("=" * 60)

    print(f"\nScenarios (sample of {min(5, len(all_scenarios))}):")
    for scenario in all_scenarios[:5]:
        print(f"  - {scenario}")

    print(f"\nPlot Twists (sample of {min(5, len(all_plot_twists))}):")
    for twist in all_plot_twists[:5]:
        print(f"  - {twist}")

    print(f"\nArchetypes (sample of {min(5, len(all_archetypes))}):")
    for archetype in all_archetypes[:5]:
        print(f"  - {archetype}")

    print("\n" + "=" * 60)
    print(" Template creation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Expand templates manually or with LLM")
    print(f"  2. Edit files in: {TEMPLATES_DIR}")
    print(f"  3. Set up fine-tuning: notebooks/fine_tune_llama.ipynb")


if __name__ == "__main__":
    main()
