from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import debugpy
import os
import random
import logging
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, WebDriverException

# Attach the debugger only if DEBUGPY=1 in the environment
if os.environ.get("DEBUGPY", "0") == "1":
    print("⏳ Waiting for debugger attach...")
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()
    print("✅ Debugger attached!")

# Config section
REPORT_OPTION_XPATH = "/html/body/div[1]/div/main/div/div/div/div[2]/div/div[1]/div/div/div[2]/div[2]/button"
ALL_DATA_XPATH = "/html/body/div[3]/div/div[2]/div/div[2]/div[2]/form/div[1]/div[1]/div/div[1]/div[2]/div/div/div/div/div/div/div[6]"
APPLY_REPORT_XPATH = "/html/body/div[3]/div/div[2]/div/div[2]/div[2]/form/div[3]/div[2]/div/div/div/div/div/button"
PANEL_ID = "SalesTransaction-panel-sales"
DROPDOWN_CLASS = "ant-pagination-options"
OPTION_XPATH = ".//div[contains(@class, 'ant-select-item-option') and contains(., '100 / page')]"
NEXT_BTN_CLASS = "ant-pagination-next"
NEXT_BTN_CSS = "li.ant-pagination-next"
MAX_RETRIES = 3
WAIT_TIME = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Set up the WebDriver
service = Service("/path/to/chromedriver")  # Update with the path to your chromedriver
def get_driver() -> webdriver.Remote:
    """Create and return a Selenium Remote WebDriver instance."""
    return webdriver.Remote(
        command_executor="http://selenium:4444/wd/hub", options=chrome_options
    )

def scrape_data(url: str, driver: webdriver.Remote) -> list:
    """Scrape property data from a given EdgeProp SG URL using the provided driver."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Scraping data from {url} (attempt {attempt})")
            driver.get(url)
            wait = WebDriverWait(driver, WAIT_TIME)
            logger.debug(f"Page loaded: {driver.current_url}, page source length: {len(driver.page_source)}")
            # Press Report Option button
            try:
                report_option_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, REPORT_OPTION_XPATH))
                )
                report_option_button.click()
                logger.info("Report Option button clicked")
            except TimeoutException:
                logger.error("Timeout waiting for Report Option button.")
                raise
            time.sleep(1)
            # Click on "All Data" button
            try:
                all_data_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, ALL_DATA_XPATH))
                )
                all_data_button.click()
                logger.info("All Data button clicked")
            except TimeoutException:
                logger.error("Timeout waiting for All Data button.")
                raise
            time.sleep(random.uniform(1, 2))
            # Click on "Apply Report Options" button
            try:
                apply_report_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, APPLY_REPORT_XPATH))
                )
                apply_report_button.click()
                logger.info("Apply Report Options button clicked")
            except TimeoutException:
                logger.error("Timeout waiting for Apply Report Options button.")
                raise
            time.sleep(random.uniform(2, 4))
            # Locate the dropdown panel
            try:
                panel = wait.until(EC.presence_of_element_located((By.ID, PANEL_ID)))
                dropdown = panel.find_element(By.CLASS_NAME, DROPDOWN_CLASS)
                wait.until(EC.element_to_be_clickable(dropdown))
                time.sleep(random.uniform(1, 2))
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", dropdown)
                dropdown.click()
                logger.info("Dropdown clicked")
            except TimeoutException:
                logger.error("Timeout waiting for dropdown panel.")
                raise
            # Wait for the options list to load
            try:
                option = wait.until(EC.presence_of_element_located((By.XPATH, OPTION_XPATH)))
                driver.execute_script(
                    """
                    arguments[0].setAttribute('aria-selected', 'true');
                    arguments[0].dispatchEvent(new MouseEvent('click', {bubbles: true}));
                    """,
                    option,
                )
                logger.info("100 / page option selected")
            except TimeoutException:
                logger.error("Timeout waiting for 100 / page option.")
                raise
            time.sleep(random.uniform(1, 2))
            data = []
            count = 0
            while True:
                count += 1
                try:
                    logger.debug(f"Scraping page {count} of {url}")
                    panel = wait.until(EC.presence_of_element_located((By.ID, PANEL_ID)))
                    rows = panel.find_elements(By.CSS_SELECTOR, ".ant-table-tbody > tr")
                    logger.debug(f"Found {len(rows)} rows on page {count}")
                    if not rows or len(rows) < 2:
                        logger.warning(f"No data rows found in table for {url} (page {count}). Saving HTML for debug.")
                        save_failed_html(driver.page_source, url, str(count))
                        driver.save_screenshot(f"data/failed_html/{url.replace('https://', '').replace('/', '_')}_page{count}.png")
                        if count == 1:
                            raise RuntimeError(f"No data rows found in table for {url} (page {count}) on first page.")
                        else:
                            logger.info("No more data rows (end of pagination). Breaking loop.")
                            break
                    first_row_text = rows[1].text if len(rows) > 1 else None
                    logger.debug(f"First row text: {first_row_text}")
                    for row in rows[1:]:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        row_data = [cell.text.strip() for cell in cells]
                        data.append(row_data)
                    # Always re-find the next button just before clicking
                    panel = wait.until(EC.presence_of_element_located((By.ID, PANEL_ID)))
                    next_button = panel.find_element(By.CLASS_NAME, NEXT_BTN_CLASS)
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, NEXT_BTN_CSS)))
                    next_btn_class = next_button.get_attribute("class")
                    logger.debug(f"Next button class: {next_btn_class}")
                    if not next_btn_class:
                        logger.info("Next button is not enabled.")
                        break
                    if next_btn_class and "ant-pagination-disabled" in next_btn_class:
                        logger.info("No more pages to navigate.")
                        break
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(random.uniform(1, 2))
                    for _ in range(20):  # up to 4 seconds
                        panel = wait.until(EC.presence_of_element_located((By.ID, PANEL_ID)))
                        new_rows = panel.find_elements(By.CSS_SELECTOR, ".ant-table-tbody > tr")
                        new_first_row_text = new_rows[1].text if len(new_rows) > 1 else None
                        if new_first_row_text != first_row_text:
                            break
                        time.sleep(0.2)
                    logger.info(f"Next button clicked for page {count}")
                except (TimeoutException, StaleElementReferenceException, WebDriverException) as e:
                    logger.error(f"Exception in pagination loop for {url} (page {count}): {type(e).__name__}: {e}")
                    save_failed_html(driver.page_source, url, f'exception_page{count}')
                    driver.save_screenshot(f"data/failed_html/{url.replace('https://', '').replace('/', '_')}_exception_page{count}.png")
                    if "No more pages" in str(e) or "ant-pagination-disabled" in str(e):
                        logger.info("No more pages or an error occurred: %s", e)
                        break
                    raise
            if not data:
                logger.error(f"No data extracted for {url}. Saving HTML for debug.")
                save_failed_html(driver.page_source, url, 'final')
                driver.save_screenshot(f"data/failed_html/{url.replace('https://', '').replace('/', '_')}_final.png")
            logger.info(f"Scraping completed. Total pages scraped: {count}")
            return data
        except (TimeoutException, StaleElementReferenceException, WebDriverException, RuntimeError) as e:
            logger.error(f"Exception during scraping {url} (attempt {attempt}): {type(e).__name__}: {e}")
            if hasattr(driver, 'page_source'):
                save_failed_html(driver.page_source, url, f'exception{attempt}')
                driver.save_screenshot(f"data/failed_html/{url.replace('https://', '').replace('/', '_')}_exception{attempt}.png")
            if attempt == MAX_RETRIES:
                logger.fatal(f"Failed to scrape {url} after {MAX_RETRIES} attempts.")
                return []
            logger.info("Retrying...")
            time.sleep(random.uniform(2, 5))
    return []

def save_failed_html(html: str, url: str, page: str) -> None:
    """Save the raw HTML of a failed page for debugging."""
    safe_url = url.replace('https://', '').replace('/', '_').replace(':', '_')
    failed_dir = os.path.join(os.getcwd(), 'data', 'failed_html')
    os.makedirs(failed_dir, exist_ok=True)
    filename = f"{safe_url}_page{page}.html"
    filepath = os.path.join(failed_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.debug(f"Saved failed HTML to {filepath}")

def main() -> None:
    """Main entry point for scraping EdgeProp SG property data."""
    driver = None
    try:
        driver = get_driver()
        urls = [
            # Add your URLs here
            "https://www.edgeprop.sg/condo-apartment/parc-vista",
            # ...
        ]
        cols = [
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
        for url in urls:
            rows = scrape_data(url, driver)
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows, columns=cols)
                last_split = url.split("/")[-1]
                df.to_excel(f"{last_split}.xlsx", index=False)
                logger.info(f"Data saved to {last_split}.xlsx")
            else:
                logger.warning(f"No data scraped for {url}")
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver closed.")

if __name__ == "__main__":
    main()
