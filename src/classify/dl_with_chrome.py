import os
import time

import logging

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from classify.agents import getRandomAgent

load_dotenv()
DOWNLOAD_DIR = os.getenv("CHROME_DOWNLOAD_DIR")
logger = logging.getLogger(__name__)


def handle_download(initial_files,  url):
    download_dir = DOWNLOAD_DIR
    new_file = None
    timeout = 10  # Max time to wait for a download to finish
    start_time = time.time()
    while True:
        current_files = set(os.listdir(download_dir))
        new_files = current_files - initial_files
        if new_files:
            # first element of the set:
            test = next(iter(new_files))
            if test.startswith(".com.google") or test.endswith("crdownload") and not test.endswith("pdf"):
                # Chrome seems to use varion names for partial downloads
                # So wait for the next file appear which may or may not end in pdf
                time.sleep(1)
                continue
            new_file = new_files.pop()
            
            break
        elif time.time() - start_time > timeout:

            break
        else:
            time.sleep(1)  # Check every second for a new file

    if new_file:
        new_file = os.path.join(download_dir, new_file)
        return new_file

    else:
        logger.info(f"Failed to download {url}")
        return None


def download_file(url):
    download_path = DOWNLOAD_DIR
    # Set up Firefox profile to handle downloads automatically

    # Set up Firefox options
    options = Options()
    options.add_argument("--headless")
    options.add_argument('--log-level=3')

    # User-Agent
    options.add_argument(f"user-agent={getRandomAgent()}")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Download Preferences
    prefs = {
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "download.default_directory": download_path,
        "plugins.always_open_pdf_externally": True  # Disables Chrome PDF Viewer
    }

    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    # driver.set_page_load_timeout(5)
    #
    # Navigate to URL and initiate download
    initial_files = set(os.listdir(download_path))
    try:
        driver.get(url)
        new_file = handle_download(initial_files, url)
            
        
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        driver.quit()
        return ""
    if not new_file:
        logger.error(f"Failed to download {url}")
        # lets take a peak at what they sent us
        doc_text =  driver.find_element(By.TAG_NAME, "body").text
        print('take a peak')
        if doc_text:
            print("Anything in the doc_text?", doc_text[:100])
        driver.quit()
        return ""
    # Close the driver
    driver.quit()
    return new_file

