"""
Filter fragments of stories that are self-contained. This means the fragment is a story itself, not a random part of a bigger story. 

This script uses SemAxis (manual seed stories) to filter out content that is too fragmented or random. 
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import PROCESSED_DATA_DIR

def get_semaxis(df): 
    """
    Use manually selected stories as semaxis seeds. 
    Returns embedding for seeds (self-contained vs fragmented stories)
    """
    # get semaxis poles 
    self_contained_ids = [
        # IDs of self-contained stories
        "ao3_38714697_1266",
        "ao3_29548224_68",
        "ao3_44979484_7460",
        "ao3_20062966_26",
        "ao3_25085842_1104",
        "ao3_19042240_957",
        "ao3_453579_399",
        "ao3_25085842_2119",
        "ao3_40307019_350",
        "ao3_1117942_1950",
        "ao3_34259563_35",
        "ao3_19724044_34",
        "ao3_53634157_594",
        "ao3_15146219_296",
        "ao3_34816549_11489",
        "ao3_34500952_1802",
        "ao3_5363684_3751",
        "ao3_20463974_1161",
        "ao3_26364097_3644",
        "ao3_8245553_50",
        "ao3_34816549_5796",
        "ao3_23026369_505",
        "ao3_3171550_1025",
        "ao3_25757818_397",
        "ao3_39597543_181",
        "ao3_3592992_50",
        "ao3_47193232_33",
        "ao3_46351063_1318",
        "ao3_25588000_200",
        "ao3_22942099_1001",
        "ao3_28777140_1078",
        "ao3_17946929_680",
        "ao3_26364097_3487",
        "ao3_14707098_307",
        "ao3_30812927_339",
        "ao3_39842187_1054",
        "ao3_22942099_4237",
        "ao3_39492981_821",
        "ao3_7127210_1509",
        "ao3_30349320_2155",
        "ao3_4034197_624",
        "ao3_25085842_1699",
        "ao3_30652973_2799",
        "ao3_8738770_185", 
        "ao3_34816549_381", 
        "ao3_42084009_583", 
        "ao3_24376396_381", 
        "ao3_26364097_2659",
        "ao3_22942099_2467",
        "ao3_40146012_217",
        "ao3_32163913_1448",
        "ao3_28777140_5256"
    ]

    fragmented_ids = [
        # IDs of fragmented/mid-conversation snippets 
        "ao3_22038418_1038",
        "ao3_29901264_167",
        "ao3_39735819_534",
        "ao3_8909155_25",
        "ao3_32163913_1712",
        "ao3_30812927_403",
        "ao3_42910668_42",
        "ao3_10643571_3347",
        "ao3_46351063_450",
        "ao3_26523892_6249",
        "ao3_25588000_531",
        "ao3_34816549_11384",
        "ao3_34577035_1813",
        "ao3_53735629_372",
        "ao3_16392173_9244",
        "ao3_35515399_961",
        "ao3_1046159_0",
        "ao3_34577035_3176",
        "ao3_53813671_452",
        "ao3_52884502_85",
        "ao3_16392173_5400",
        "ao3_26523892_3270",
        "ao3_8300672_434",
        "ao3_234222_167",
        "ao3_19334392_2533",
        "ao3_41199483_1082",
        "ao3_34816549_12520",
        "ao3_26364097_9398",
        "ao3_12805206_2467",
        "ao3_24773944_381",
        "ao3_10588629_590",
        "ao3_34816549_9088",
        "ao3_19110040_494",
        "ao3_30349320_317",
        "ao3_34586890_1637",
        "ao3_1035300_404",
        "ao3_34500952_1530",
        "ao3_30153540_578",
        "ao3_17946929_972",
        "ao3_24273403_18",
        "ao3_25085842_1460",
        "ao3_16392173_2172",
        "ao3_30349320_3467",
        "ao3_50510668_1660",
        "ao3_10643571_3306",
        "ao3_12025527_792",
        "ao3_7569610_32",
        "ao3_30547047_326",
        "ao3_12402426_841",
        "ao3_34577035_3451",
        "ao3_34816549_502",
        "ao3_1117942_3618",
        "ao3_44979484_1886",
        "ao3_41199483_74",
        "ao3_34816549_5124",
        "ao3_44979484_8135",
        "ao3_13867242_364"
    ]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    self_contained_texts = df[df['id'].isin(self_contained_ids)]['text'].tolist()
    fragmented_texts = df[df['id'].isin(fragmented_ids)]['text'].tolist()

    self_contained_embeddings = model.encode(self_contained_texts, show_progress_bar=True)
    fragmented_embeddings = model.encode(fragmented_texts, show_progress_bar=True)

    self_contained_avg = np.mean(self_contained_embeddings, axis=0)
    fragmented_avg = np.mean(fragmented_embeddings, axis=0)

    semaxis = self_contained_avg - fragmented_avg

    return semaxis

def get_all_embeddings(df): 
    """
    Generate sentence embeddings for all sentences. 
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_embeddings = model.encode(df['text'].tolist(), 
                                  batch_size=256, 
                                  show_progress_bar=True)
    
    return all_embeddings

def main():
    print("Reading in parquet...")
    fragments_path = PROCESSED_DATA_DIR / "ao3_fragments.parquet"
    df = pd.read_parquet(fragments_path)

    # create semaxis pole 
    print("Creating semaxis pole")
    semaxis = get_semaxis(df)

    # get sentence embeddings for entire df 
    print("Getting sentence embeddings for all")
    all_embeddings = get_all_embeddings(df)

    # calculate cosine similarity for all fragments 
    print("Calculating cosine similarity")
    scores = cosine_similarity(all_embeddings, semaxis.reshape(1, -1)).flatten()
    df['semaxis_score'] = scores 

    # filter out fragments by semaxis threshold 
    threshold = 0.0 
    filtered_df = df[df['semaxis_score'] >= threshold].copy()
    filtered_df = filtered_df.sort_values(by='semaxis_score', ascending=False)

    # save df
    print("Saving DF")
    output_path = PROCESSED_DATA_DIR / "ao3_fragments_selfcontained.parquet"

    filtered_df.to_parquet(output_path, index=False)
    print(f"Saved {len(filtered_df)} rows successfully!")

if __name__ == "__main__":
    main()
