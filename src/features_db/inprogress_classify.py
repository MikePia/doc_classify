# %%
import logging
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
logging.basicConfig(filename="document_processing_errors.log", level=logging.INFO)

# %% [markdown]
#


# %%
def pdf_to_text(path):
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            try:
                text += page.get_text()
            except Exception as page_error:
                print(f"Error extracting text from page in {path}: {page_error}")
                continue
                # Optionally, continue to the next page or log the error
        return text
    except Exception as e:
        logging.info(f"Error processing file {path}: {e}")
        return ""


# %%

nlp = spacy.load("en_core_web_sm")  # Or a larger model as needed


def clean_and_tokenize(text, chunk_size=1000000):
    """
    Tokenizes the text using SpaCy, handling long texts by processing in chunks.

    :param text: The text to be tokenized.
    :param chunk_size: Maximum chunk size in characters.
    :return: A string of the lemmatized tokens.
    """
    # Check if the text length exceeds the chunk size
    if len(text) > chunk_size:
        # Initialize an empty list to store tokens from all chunks
        tokens_all_chunks = []

        # Process the text in chunks
        for start in range(0, len(text), chunk_size):
            end = start + chunk_size
            # Extract a chunk of text
            chunk = text[start:end]
            # Process the chunk
            doc = nlp(chunk)
            # Extract tokens, lemmatize, and filter as before
            tokens = [
                token.lemma_ for token in doc if token.is_alpha and not token.is_stop
            ]
            tokens_all_chunks.extend(tokens)

        # Combine tokens from all chunks and return
        return " ".join(tokens_all_chunks)
    else:
        # If text does not exceed the chunk size, process as before
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return " ".join(tokens)


# Example of how to apply this function to your DataFrame
# df['tokenized_text'] = df['cleaned_text'].apply(clean_and_tokenize)


# %% [markdown]
# # 3. Feature Extraction
# - **Combine keyword-matching and TF-IDF**
# - **TF-IDF Vectorization:** Use Scikit-learn's TfidfVectorizer to convert the cleaned text documents into a matrix of TF-IDF features.


# %%
# Function to check for keyword presence
def check_keywords(text, keyword_list):
    text = text.lower()
    return int(any(keyword in text for keyword in keyword_list))


# %% [markdown]
# # alternate code below


# %%
def combine_tfidf_keyword(df):
    # Step 2: TF-IDF Calculation
    vectorizer = TfidfVectorizer(
        max_features=5000
    )  # Adjust number of features as needed
    tfidf_matrix = vectorizer.fit_transform(df["tokenized_text"])

    # Step 3. Combine keyword and tfidf features into a single matrix
    # Convert binary keyword matches to a matrix
    keyword_features = df[[col for col in df.columns if "_keyword" in col]].to_numpy()
    # Combine TF-IDF features with keyword binary indicators
    combined_features = np.hstack((tfidf_matrix.toarray(), keyword_features))

    # Now `combined_features` is ready for model training, and should be aligned with your labels.
    return combined_features


# %% [markdown]
# # 4. Training the Classification Model
#
# ### Next Steps
#
#     Train Your Model: Use the combined_features matrix along with your labels to train and evaluate your classification model.
#     Evaluation and Refinement: Assess the model's performance and adjust your keyword lists, TF-IDF parameters, or model choice as needed.
#
# This approach leverages both the specificity of keyword matching and the nuanced importance scoring of TF-IDF, providing a rich set of features for document classification.
#
# - **Splitting Data:** Use your 1000 classified documents as training data. Ensure you have a balanced dataset for the three categories.


# %%
def split_data(X, y):
    # Split the data - 70% for training, 30% for testing; adjust ratios as you see fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# %% [markdown]
# - **Model Selection and Training:** Given the textual nature of your task, models like CNN or LSTM could perform well. TensorFlow/Keras will be used here.

# %% [markdown]
# # An initial simple Binary logistic  regression


# %%
def train_logistic_regression(X_train, y_train):
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Increasing max_iter for convergence

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Return the trained model
    return model


# %% [markdown]
# # Implmentation

# %%
dataset_path = "/uw/invest-data/classify_presentations/data/dataset.csv"
DATA_PATH = "/dave/presentations/"
os.path.exists(dataset_path)
df = pd.read_csv(dataset_path, header=0)


# %%

df["fname"] = DATA_PATH + df["fname"]


# %%
df["cleaned_text"] = df["fname"].apply(pdf_to_text)


# %%
tqdm.pandas(desc="Processing documents")

df["tokenized_text"] = df["cleaned_text"].progress_apply(clean_and_tokenize)

# %%
keywords = {
    "financial_terms": [
        "financial",
        "investment",
        "share price",
        "financial metrics",
        "investment strategy",
    ],
    "legal_statements": [
        "confidentiality statement",
        "legal disclaimer",
        "disclosure statement",
        "proprietary information",
        "intellectual property",
    ],
    "company_info": [
        "company overview",
        "company analysis",
        "business model",
        "company performance",
    ],
    "presentation_content": [
        "visual aids",
        "data charts",
        "case studies",
        "comparative analysis",
    ],
    "company_targets": ["sales targets", "company targets", "performance targets"],
    "financial_discussions": [
        "financial figures",
        "financial projections",
        "financial results",
        "financial language",
    ],
    "regulatory_references": [
        "SEC filings",
        "regulatory filings",
        "external entities",
        "lawsuits",
    ],
    "detail_descriptions": [
        "loan details",
        "product details",
        "research and development",
        "financial details",
    ],
    "company_specific": [
        "company specific",
        "industry specific",
        "company-specific analysis",
        "specific company focus",
    ],
    # "Other Clusters" category is omitted since it's broad and without specific keywords
}


# Apply keyword matching
for category, keyword_list in keywords.items():
    df[category + "_keyword"] = df["tokenized_text"].apply(
        check_keywords, args=(keyword_list,)
    )


# %%
features = combine_tfidf_keyword(df)


# %%
X_train, X_test, y_train, y_test = split_data(features, df["presentation"])

# %%
model = train_logistic_regression(X_train, y_train)

# %%
# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate and print the evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="binary"))
print("Recall:", recall_score(y_test, y_pred, average="binary"))
print("F1 Score:", f1_score(y_test, y_pred, average="binary"))
