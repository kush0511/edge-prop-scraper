# edge-prop-scraper

Scrapes property transaction data from the [EdgeProp SG website](https://www.edgeprop.sg/) and performs analysis.

## Quickstart: Docker Compose (Recommended)

1. **Build and start all services:**
   ```sh
   docker compose up --build
   ```
   This will launch the Selenium server and run the scraper/analysis pipeline automatically.

2. **Configure which condos to scrape:**
   - Edit the `URL_SUFFIXES` environment variable in `compose.yaml` to a comma-separated list (e.g. `parc-vista,caspian,the-lakeshore`).
   - By default, only `parc-vista` is scraped.

3. **View results:**
   - Scraped Excel data: `scraped_data/all_properties.xlsx`
   - Analysis/plots: `results/`
   - Failed HTML/screenshots: `data/failed_html/`

> **Note:** By default, the scraper will not wait for a debugger to attach. To enable debugging (and pause at startup), set `DEBUGPY=1` in the environment variables in `compose.yaml`.

---

## Manual Usage

### 1. Install dependencies
```sh
pip install -r requirements.txt
```

### 2. Set up Selenium
- Make sure you have a Selenium server running (e.g., via Docker Compose)
- Update the path to your `chromedriver` in `scraper.py` if running locally

### 3. Run the Scraper
```sh
python scraper.py
```
- Output Excel files will be saved for each project.
- Failed HTML/screenshots are saved in `data/failed_html/` for debugging.

### 4. Run the Analysis
```sh
python main.py
```
- This will process the scraped data and generate analysis/plots in the `results/` folder.

## Configuration
- All XPaths, CSS selectors, and retry/sleep parameters are at the top of `scraper.py` for easy adjustment.
- The list of condos to scrape is controlled by the `URL_SUFFIXES` environment variable (see Docker section above).

## Notes
- The scraper uses headless Chrome by default.
- Logging is used throughout for robust diagnostics.
- WebDriver is always closed after use.

## Troubleshooting
- If scraping fails, check the `data/failed_html/` folder for HTML and screenshots of the error state.
- Adjust sleep times or selectors in `scraper.py` if the site layout changes.

---

## Technical Design

### Overview
This project is a robust, modular pipeline for scraping and analyzing Singapore condo transaction data from [EdgeProp SG](https://www.edgeprop.sg/). It is designed for reliability, maintainability, and ease of deployment in production or research settings.

### Architecture
- **Scraper (`scraper.py`)**: Uses Selenium with headless Chrome to extract transaction tables from EdgeProp SG. All selectors, timeouts, and retry logic are configurable at the top of the file. The scraper:
  - Handles anti-bot measures with random delays and retries.
  - Catches and logs Selenium exceptions, saving failed HTML and screenshots for debugging.
  - Outputs data as Excel files for each project.

- **Main Orchestration (`main.py`)**: Controls the end-to-end workflow:
  - Reads the list of condos to scrape from the `URL_SUFFIXES` environment variable (set in Docker Compose or manually).
  - Iterates through each project, scrapes data, processes it, and merges all results.
  - Saves the combined dataset and triggers the analysis pipeline.
  - Uses logging throughout and ensures all resources are cleaned up.

- **Analysis (`property_analysis.py`)**: Provides a professional, extensible analysis pipeline:
  - Cleans and validates the scraped data.
  - Computes price indices, annual/quarterly returns, and performance transitions.
  - Generates plots and summary statistics, saving all results to the `results/` folder.

- **Dockerization**: The project is fully containerized for reproducibility:
  - `Dockerfile` builds the Python environment.
  - `compose.yaml` launches both the scraper/analysis pipeline and a Selenium server.
  - All configuration (e.g., which condos to scrape) is controlled via environment variables.

### Output Structure
- `scraped_data/`: Combined Excel file of all scraped transactions.
- `results/`: All analysis outputs and plots.
- `data/failed_html/`: HTML and screenshots of failed scraping attempts for debugging.

### Scraper Design Details

The scraping logic in `scraper.py` is engineered to handle the unique challenges of the EdgeProp SG website, which generates reports that can take a long time to load and present data in a highly dynamic, interactive table. Key design considerations include:

- **Waiting for Full Page Load:**
  - Reports on EdgeProp SG can take a significant amount of time to generate, especially for large projects or long time spans. The scraper uses explicit waits to ensure all required elements (tables, menus, buttons) are fully loaded before attempting to interact with them. This minimizes the risk of missing data due to partial loads.

- **Navigating Menus and Report Options:**
  - To access all historical transaction data, the scraper must open the report options menu and change the time span to the maximum available. This involves simulating user interactions to open dropdowns, select date ranges, and apply filters, all while waiting for the UI to update.

- **Extracting Data from Dynamic Tables:**
  - The transaction data is presented in a paginated, JavaScript-rendered table. By default, only 10 rows are shown per page. The scraper programmatically changes this to 100 rows per page to minimize the number of pagination actions required.
  - It then iterates through all available pages, extracting the data from each, and combines the results into a single dataset.

- **Handling Popups and UI Interruptions:**
  - The EdgeProp site occasionally displays random popups (e.g., chatbots, contact us prompts) that can obscure buttons or table elements. These popups can cause Selenium actions to fail or elements to be unclickable.
  - To address this, the scraper implements a retry policy: if any Selenium action fails (due to popups or slow loads), it retries the operation several times, with random delays between attempts. This increases the likelihood of success even in the presence of transient UI issues.

- **Error Handling and Debugging:**
  - If scraping fails after all retries, the scraper saves the current HTML and a screenshot of the page for post-mortem debugging. This helps diagnose issues such as site layout changes or new types of popups.

- **Configurability:**
  - All XPaths, CSS selectors, wait times, and retry parameters are defined at the top of `scraper.py` for easy adjustment if the site changes.

This design ensures that the scraper is resilient to the slow, dynamic, and sometimes unpredictable nature of the EdgeProp SG website, maximizing the chances of extracting complete and accurate data for analysis.

---

**Author:** Kushal Sai Gunturi
