"""
AO3 (Archive of Our Own) scraper for collecting stories
Note: AO3 allows respectful scraping. Be sure to respect rate limits (3-5 seconds between requests)
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import time
from pathlib import Path
import sys
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    AO3_TAGS,
    AO3_MAX_WORKS,
    RAW_DATA_DIR
)

class AO3Scraper:
    """Scraper for AO3 fanfiction stories"""

    def __init__(self, max_stories_per_tag=10):
        """Initialize AO3 scraper"""
        self.base_url = "https://archiveofourown.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.stories = []
        self.seen_ids = set()  # Track IDs across all tags to avoid duplicates
        self.max_stories_per_tag = max_stories_per_tag
        self.rate_limit_delay = 3  # AO3 rate limit: 3 seconds between requests

    def count_words(self, text):
        """Count words in text"""
        return len(text.split())

    def search_by_tag(self, tag, max_stories=None):
        """Search AO3 by tag and collect up to max_stories unique stories"""
        if max_stories is None:
            max_stories = self.max_stories_per_tag
            
        print(f"\nSearching AO3 for tag: '{tag}' (target: {max_stories} unique stories)...")

        collected_count = 0
        skipped_duplicates = 0
        page = 1
        max_pages = 10  # Safety limit to prevent infinite loops

        while collected_count < max_stories and page <= max_pages:
            # Build tag URL - use tag browsing instead of search
            # Format: https://archiveofourown.org/tags/Humor/works
            tag_encoded = tag.replace(' ', '%20').replace('/', '*s*')
            tag_url = f"{self.base_url}/tags/{tag_encoded}/works"
            
            params = {
                'view_adult': 'true', 
                'commit': 'Sort and Filter',
                'work_search[language_id]': 'en',
                'work_search[complete]': 'T',
                'work_search[sort_column]': 'kudos_count',
                'page': page
            }

            # Retry logic for timeouts
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.session.get(tag_url, params=params, timeout=60)
                    response.raise_for_status()
                    break  # Success, exit retry loop
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"Timeout! Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed after {max_retries} attempts. Moving to next tag.")
                        return collected_count
                        
                except Exception as e:
                    print(f"Error: {e}")
                    return collected_count

            try:
                # DEBUG: Check response
                print(f"  Response status: {response.status_code}")
                print(f"  Response length: {len(response.content)} bytes")
                print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")

                soup = BeautifulSoup(response.content, 'html.parser')

                # DEBUG: Check what we're getting
                title_tag = soup.find('title')
                if title_tag:
                    print(f"  Page title: {title_tag.text.strip()}")
                else:
                    print(f"  NO TITLE TAG FOUND - HTML might be broken")
                    print(f"  First 500 chars of response: {response.text[:500]}")

                # Find all work items - FIXED: Use lambda to match elements with all three classes
                work_items = soup.find_all('li', class_=lambda x: x and all(cls in x for cls in ['work', 'blurb', 'group']))

                # DEBUG: Check for any li elements
                all_li = soup.find_all('li')
                print(f"  Total <li> elements found: {len(all_li)}")
                print(f"  Work items found: {len(work_items)}")

                if not work_items:
                    print(f"No more works found on page {page}")
                    break

                print(f"Found {len(work_items)} works on page {page}")

                for work in work_items:
                    if collected_count >= max_stories:
                        break
                        
                    try:
                        # Extract work ID and URL
                        work_id = work.get('id', '').replace('work_', '')
                        if not work_id:
                            continue

                        # Skip if already seen this ID across any tag
                        if work_id in self.seen_ids:
                            skipped_duplicates += 1
                            print(f"Skipping duplicate (already seen in another tag)")
                            continue

                        # FIXED: Add view_full_work=true to get entire story
                        work_url = f"{self.base_url}/works/{work_id}?view_adult=true&view_full_work=true"

                        # Get work details
                        story_data = self.scrape_work(work_url, work_id, tag)

                        if story_data:
                            self.stories.append(story_data)
                            self.seen_ids.add(work_id)
                            collected_count += 1
                            print(f"Collected {collected_count}/{max_stories} for '{tag}'")

                        # Respect rate limit
                        time.sleep(self.rate_limit_delay)

                    except Exception as e:
                        print(f"Error processing work {work_id}: {e}")
                        continue

                # Check if we need to continue to next page
                if collected_count < max_stories:
                    print(f"Moving to page {page + 1}...")
                    time.sleep(self.rate_limit_delay)
                    page += 1
                else:
                    break

            except Exception as e:
                print(f"Error processing page {page}: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Collected {collected_count} unique stories for '{tag}'")
        if skipped_duplicates > 0:
            print(f"(Skipped {skipped_duplicates} duplicates)")
        return collected_count

    def scrape_work(self, work_url, work_id, tag):
        """Scrape a single work from AO3"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(work_url, timeout=60)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Get title
                title_elem = soup.find('h2', class_='title')
                title = title_elem.text.strip() if title_elem else "Unknown"

                # Get maturity rating
                maturity_rating = "Not Rated"
                rating_elem = soup.find('dd', class_='rating tags')
                if rating_elem:
                    rating_link = rating_elem.find('a')
                    if rating_link:
                        maturity_rating = rating_link.text.strip()
                
                # Determine if NSFW based on maturity rating
                nsfw = maturity_rating in ["Mature", "Explicit"]

                # Get story text - this now includes ALL chapters
                story_div = soup.find('div', id='chapters')
                if not story_div:
                    return None

                # Extract text from all paragraphs
                paragraphs = story_div.find_all('p')
                story_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])

                if not story_text:
                    return None

                # Get metadata
                stats_elem = soup.find('dl', class_='stats')
                kudos = 0

                if stats_elem:
                    # Get kudos count
                    kudos_elem = stats_elem.find('dd', class_='kudos')
                    if kudos_elem:
                        kudos_text = kudos_elem.text.strip()
                        kudos = int(re.sub(r'[^\d]', '', kudos_text)) if kudos_text else 0

                # Get tags
                tags = []
                tag_elems = soup.find_all('dd', class_='freeform tags')
                for tag_li in tag_elems:
                    tag_links = tag_li.find_all('a')
                    for tag_link in tag_links:
                        if tag_link:
                            tags.append(tag_link.text.strip())

                return {
                    "id": f"ao3_{work_id}",
                    "text": story_text,
                    "title": title,
                    "maturity_rating": maturity_rating,
                    "nsfw": nsfw,
                    "word_count": self.count_words(story_text),
                    "kudos": kudos,
                    "tags": tags,
                    "search_tag": tag,
                    "source": "ao3",
                    "url": work_url,
                }

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"  Timeout on work {work_id}, retrying...")
                    time.sleep(3)
                else:
                    print(f"  Failed to fetch work {work_id} after {max_retries} attempts")
                    return None
                    
            except Exception as e:
                print(f"  Error scraping work {work_id}: {e}")
                return None
        
        return None

    def get_stats_by_tag(self):
        """Get count of stories collected per tag"""
        tag_counts = {}
        for story in self.stories:
            tag = story.get('search_tag', 'Unknown')
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

def scrape_all_tags(max_stories_per_tag=10): 
    """
    Scrape for all defined tags.
    """
    scraper = AO3Scraper(max_stories_per_tag=max_stories_per_tag)

    print(f"Starting full scrape...")
    print()

    # Scrape for each tag
    for i, tag in enumerate(AO3_TAGS):
        print(f"[{i+1}/{len(AO3_TAGS)}] Processing tag: '{tag}'\n")
        scraper.search_by_tag(tag)

    print(f"Total unique stories collected: {len(scraper.stories)}")

    return scraper 

def main():
    """Main scraping function"""
    scraper = scrape_all_tags()

    # save 
    output_path = RAW_DATA_DIR / "ao3_stories.csv"
    all_stories = pd.DataFrame(scraper.stories)
    all_stories.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
