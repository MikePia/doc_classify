# Copyright (c) 2024 ZeroSubstance. All rights reserved.

# This software and its content are licensed under the Software License Agreement
# provided by ZeroSubstance. Use of this software for any non-commercial purpose is
# permitted under the terms of the license. Commercial use is prohibited without
# a separate license agreement with ZeroSubstance. For license inquiries, contact
# lynnpete@proton.me.

import os

import classify.loggingsetup as lsetup
import logging
import time
import pandas as pd
import csv

from classify.preprocess import clean_and_tokenize, pdf_to_text
from classify.extract_features import (
    extract_specific_features,
    load_from_disk,
    keywords,
    check_keywords,
    combine_tfidf_keyword_additional_features,
)
from dotenv import load_dotenv
from classify.dl_with_chrome import download_file
from classify.dl_with_firefox import download_file as download_file_firefox

print(lsetup.x)
logger = logging.getLogger(__name__)
load_dotenv()

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


def predict_examples():
    # fn = "/dave/presentations/WBA-3Q-2023-Presentation.pdf"
    fn = "/home/mike/Downloads/evolution-advice-subcommittee-presentation-102821(1).pdf"
    if not os.path.exists(fn):
        raise ValueError(f"File not found: {fn}")

    # For a single prediction from each model
    start = time.time()
    doc_feature_array = prepare_single_document(fn, tdif_vectorizer)
    # doc_feature_array.shape
    # single_prediction_model1 = model1.predict(doc_feature_array)
    # print("Random Forest Prediction:", single_prediction_model1)
    # print("Time taken for single prediction:", time.time() - start)

    # # %%

    single_prediction_model2 = model2.predict(doc_feature_array)
    print("HistGradientBoosting Prediction:", single_prediction_model2)
    print("Time taken:", time.time() - start)

    # # %%
    # start = time.time()
    # single_prediction_model3 = model3.predict(doc_feature_array)
    # print("CatBoost Prediction:", single_prediction_model3)
    # print("Time taken:", time.time() - start)

    # # %%
    # start = time.time()
    # dtest = xgb.DMatrix(doc_feature_array)

    # single_prediction_model4 = model4.predict(dtest)
    # print("XGBoost Prediction:", single_prediction_model4)

    # print("Time taken:", time.time() - start)

    # # %% [markdown]
    # # # Working on processing links in memory here

    # # %%


# Open dataset.csv as dataframe
def open_dataset(path):
    df = pd.read_csv(path)
    return df

def download_it(url):
    new_file = download_file(url)
    if new_file:
        return new_file
    new_file = download_file_firefox(url)
    if new_file:
        return new_file
    return ""


def main():
    download_dir = "/dave/tmp"
    dataset_path = "/uw/invest-data/classify_presentations/data/dataset.csv"
    dataset_df = open_dataset(dataset_path)
    predictions = []
    start = time.time()
    # iterate through the rows 17-25 of the dataset
    processed_files = 'data/processed_files_' + str(start) + '.csv'
    if not os.path.exists(processed_files):
        with open(processed_files, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["link", "classify", "reason"])
    for i in range(101, 200):
        row = [dataset_df.iloc[i]["link"]]
        # print(dataset_df.iloc[i]["link"])
        new_file = download_it(dataset_df.iloc[i]["link"])

        if new_file:
            doc_feature_array = prepare_single_document(new_file, tdif_vectorizer)

            da_predict = model2.predict(doc_feature_array)
            row.append(da_predict)
            # remove the file from the download directory
            os.remove(os.path.join(download_dir, new_file))
            predictions.append([f'{i}: predict: {da_predict} {new_file}'])
            print(f"{i} / 100 completed type {new_file}")
        else:
            row.append("")
            row.append("Failed to Download")
            print(f"Failed to download {dataset_df.iloc[i]['link']}")
        #  Add the row to processed_files
        with open(processed_files, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    print(time.time() - start)
    for p in predictions:
        print(p)
    print()


if __name__ == "__main__":
    df, features, tdif_vectorizer, model2 = load_from_disk(include_model="hgb")
    # predict_examples()
    main()

# %%
