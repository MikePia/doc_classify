import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException
from classify.agents import getRandomAgent

DOWNLOAD_DIR = os.getenv("FIREFOX_DOWNLOAD_DIR")
logger = logging.getLogger(__name__)

def handle_download(initial_files, download_dir, url):
    # Wait for the download to start and finish
    # Adjust the time as needed based on your expected download time
    # time.sleep(2)  # Initial sleep to wait for the download to start

    # Now wait for a new file to appear in the directory
    new_file = None
    timeout = 10  # Max time to wait for a download to finish
    start_time = time.time()

    while True:
        current_files = set(os.listdir(download_dir))
        new_files = current_files - initial_files
        if new_files:
            new_file = new_files.pop()
            break
        elif time.time() - start_time > timeout:
            print(f"Timeout waiting for download to complete for {url}")
            break
        else:
            time.sleep(1)  # Check every second for a new file

    if new_file:
        new_file = os.path.join(download_dir, new_file)
        return new_file

    else:
        logger.error(f"Failed to download {url}")
        return None


def download_file(url):
    download_path = DOWNLOAD_DIR
    # Set up Firefox profile to handle downloads automatically
    gecko_driver_path = "/snap/bin/geckodriver"

    # Set up Firefox options
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.set_preference("general.useragent.override", getRandomAgent())

    # Create a Firefox Profile
    # firefox_profile = webdriver.FirefoxProfile()
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
    firefox_options.set_preference("browser.download.dir", download_path)
    firefox_options.set_preference("browser.download.useDownloadDir", True)
    firefox_options.set_preference(
        "browser.helperApps.neverAsk.saveToDisk", "application/pdf"
    )
    firefox_options.enable_downloads = True

    firefox_options.set_preference(
        "pdfjs.disabled", True
    )  # Disable Firefox's built-in PDF viewer

    # Initialize the driver with Service
    service = Service(executable_path=gecko_driver_path)
    driver = webdriver.Firefox(service=service, options=firefox_options)
    driver.set_page_load_timeout(5)
    #
    # Navigate to URL and initiate download
    initial_files = set(os.listdir(download_path))
    try:
        driver.get(url)
        handle_download(initial_files, download_path, url)
        print(f"Downloading {url} to {download_path}???")
    except TimeoutException:
        prediction = handle_download(initial_files, download_path, url)

        driver.quit()
        return prediction
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        driver.quit()
        return ""

    # Close the driver
    driver.quit()
