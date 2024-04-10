import time
from agents import getRandomAgent
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import os



# %%
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
            print(f"Timeout waiting for download to complete for {link}")
            break
        else:
            time.sleep(1)  # Check every second for a new file

    if new_file:
        new_file = os.path.join(download_dir, new_file)
        doc_feature_array = prepare_single_document(new_file, tdif_vectorizer)

        single_prediction_model2 = model2.predict(doc_feature_array)
        # remove the file from the download directory
        os.remove(os.path.join(download_dir, new_file))
        return single_prediction_model2
    else:
        logging.info(f"Failed to download {url}")
        return None




def download_file(url, download_path):
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
    
    
# %%
def prepare_single_document(filepath, tfidf_vectorizer):
    # Step 1: Convert PDF document to text
    document_text = pdf_to_text(filepath)

    # Step 2: Clean and tokenize the text
    tokenized_text = clean_and_tokenize(document_text)

    # Create a DataFrame to hold the document text
    df = pd.DataFrame({"tokenized_text": [tokenized_text]})
    df["fname"] = filepath

    # Step 3: Extract specific features (if this applies directly to the text)
    df = extract_specific_features(df)

    # Step 4: Apply keyword matching
    for category, keyword_list in keywords.items():
        df[category + "_keyword"] = df["tokenized_text"].apply(
            lambda x: check_keywords(x, keyword_list)
        )

    # Step 5: Combine TF-IDF with keyword features and any additional features
    combined_features, _ = combine_tfidf_keyword_additional_features(
        df, tfidf_vectorizer
    )

    return combined_features


if __name__ == '__main__':
    # %%
    fn = '/dave/presentations/WBA-3Q-2023-Presentation.pdf'
    os.path.exists(fn)


    # %%
    import time
    # For a single prediction from each model
    start = time.time()
    doc_feature_array = prepare_single_document(fn, tdif_vectorizer)
    doc_feature_array.shape
    single_prediction_model1 = model1.predict(doc_feature_array)
    print("Random Forest Prediction:", single_prediction_model1)
    print("Time taken for single prediction:", time.time() - start)

    # %%

    start = time.time()
    single_prediction_model2 = model2.predict(doc_feature_array)
    print("HistGradientBoosting Prediction:", single_prediction_model2)
    print("Time taken:", time.time() - start)


    # %%
    start = time.time()
    single_prediction_model3 = model3.predict(doc_feature_array)
    print("CatBoost Prediction:", single_prediction_model3)
    print("Time taken:", time.time() - start)

    # %%
    start = time.time()
    dtest = xgb.DMatrix(doc_feature_array)

    single_prediction_model4 = model4.predict(dtest)
    print("XGBoost Prediction:", single_prediction_model4)

    print("Time taken:", time.time() - start)


    # %% [markdown]
    # # Working on processing links in memory here

    # %%








