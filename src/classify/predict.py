"""Isolate prediction. Load everything needed from disk"""

import logging
import classify.loggingsetup as lsetup
import os
import pandas as pd
import time
import xgboost as xgb
from classify.util import (
    load_df_from_pickle,
    load_np_array_from_pickle,
    load_vectorizer,
    load_model,
)
from classify.preprocess import clean_and_tokenize, pdf_to_text
from classify.extract_features import (
    extract_specific_features,
    keywords,
    check_keywords,
    combine_tfidf_keyword_additional_features,
)

logger = logging.getLogger(__name__)
print(lsetup.x)


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


def load_stuff() -> tuple:
    """Load everything needed from disk
    Returns:
        tuple: (features, df, tdif_vectorizer) or None,None,None
    """
    dff_pickle_path = "/dave/data/df_features.pkl"
    features_path = "/dave/data/features_array.pkl.npy"
    # features_array_ppath = '/dave/data/features_array.pkl'
    tdif_vectorizer_pickle_path = "/dave/data/tdif_vectorizer.pkl"

    force = False
    if (
        os.path.exists(features_path)
        and os.path.exists(dff_pickle_path)
        and os.path.exists(tdif_vectorizer_pickle_path)
        and not force
    ):
        print(
            "loading dataframe with features,features numpy array, and tdif vectorizer from disk)"
        )
        features = load_np_array_from_pickle(features_path)
        df = load_df_from_pickle(dff_pickle_path)
        tdif_vectorizer = load_vectorizer(tdif_vectorizer_pickle_path)

        return features, df, tdif_vectorizer
    else:
        print("Items failed to load from disk")
        return None, None, None


if __name__ == "__main__":
    features, df, tdif_vectorizer = load_stuff()
    model1 = load_model("forest")
    model2 = load_model("hgb")
    model3 = load_model("catboost")
    model4 = load_model("xgboost")

    # For a single prediction from each model
    files = os.listdir("/dave/tmp2")
    for file in files:
        if file.endswith(".pdf"):
            fn = "/dave/tmp2/" + file

            start = time.time()
            doc_feature_array = prepare_single_document(fn, tdif_vectorizer)
            doc_feature_array.shape
            single_prediction_model1 = model1.predict(doc_feature_array)
            m2_predictg = model2.predict(doc_feature_array)
            m3_predictg = model3.predict(doc_feature_array)
            

            print("Random Forest Prediction:", single_prediction_model1)
            print("HGB Prediction:", m2_predictg)
            print("CatBoost Prediction:", m3_predictg)

            print("Time taken for single prediction:", time.time() - start)
