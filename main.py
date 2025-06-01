import logging
from typing import List
import pandas as pd
from scraper import scrape_data, get_driver
import property_analysis
import os
import re

# Config
url_suffixes = os.environ.get("URL_SUFFIXES")
if url_suffixes:
    URL_SUFFIXES = [s.strip() for s in url_suffixes.split(",") if s.strip()]
else:
    URL_SUFFIXES = ["parc-vista"]  # Default fallback
URLS = [f"https://www.edgeprop.sg/condo-apartment/{suffix}" for suffix in URL_SUFFIXES]
COLS = [
    "Date",
    "Size",
    "Bedrooms",
    "PSF",
    "Price",
    "Type of Sale",
    "Address",
    "Type of Area",
    "Purchaser Address",
    "Source",
]
SCRAPED_DATA_PATH = "./scraped_data/all_properties.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def process_scraped_dataframe(df: pd.DataFrame, url_suffix: str) -> pd.DataFrame:
    """
    Add 'Project' column, extract 'Floor' from 'Address', insert after 'Address',
    and convert 'Date' column to datetime (format: '28 MAY 2025' -> datetime).
    """
    project_name = " ".join(word.capitalize() for word in url_suffix.split("-"))
    df.insert(0, "Project", project_name)
    # Extract 'Floor' from 'Address' using regex and insert after 'Address'
    def extract_floor(address):
        # Match patterns like '#12-', '#12 ', '#12A-', '#12A ', etc.
        match = re.search(r'#(\d{1,3}[A-Za-z]?)[\s\-]', str(address))
        if match:
            floor_str = match.group(1)
            # Extract leading digits as floor number
            floor_num = re.match(r'\d+', floor_str)
            return int(floor_num.group()) if floor_num else None
        return None
    address_idx = int(df.columns.get_loc("Address")) # type: ignore
    df.insert(address_idx + 1, "Floor", df["Address"].apply(extract_floor))
    # Convert 'Date' column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
    return df

def main() -> None:
    """Orchestrate scraping and analysis."""
    driver = None
    all_dfs: List[pd.DataFrame] = []
    try:
        driver = get_driver()
        for url, url_suffix in zip(URLS, URL_SUFFIXES):
            try:
                logger.info(f"Processing {url_suffix}...")
                rows = scrape_data(url, driver)
                if not rows:
                    logger.warning(f"Skipping {url_suffix} due to empty data.")
                    continue
                df = pd.DataFrame(rows, columns=COLS)
                df = process_scraped_dataframe(df, url_suffix)
                all_dfs.append(df)
                logger.info(f"Scraped data for {url_suffix}")
            except Exception as e:
                logger.error(f"Failed to scrape {url_suffix}: {e}")
                continue
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            os.makedirs(os.path.dirname(SCRAPED_DATA_PATH), exist_ok=True)
            merged_df.to_excel(SCRAPED_DATA_PATH, index=False)
            logger.info(f"All data saved to {SCRAPED_DATA_PATH}")
            try:
                property_analysis.analyze(SCRAPED_DATA_PATH)
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
        else:
            logger.warning("No data scraped. Skipping analysis.")
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver closed.")

if __name__ == "__main__":
    main()
