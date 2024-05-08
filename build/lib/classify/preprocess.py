import logging
import os

from tqdm import tqdm
import pandas as pd
import spacy
import fitz  # PyMuPDF


logging.basicConfig(filename="preprocess_errors.log", level=logging.INFO)

logger = logging.getLogger(__name__)


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
    except fitz.FileNotFoundError:
        logger.info(f"File not found: {path}")
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        logger.info(f"Error processing file {path}: {e}")
        return ""


tqdm.pandas()  # Enables progress_apply for pandas


def update_pickle(pickle_df, dataset_path, pickle_path, data_path):
    df = pd.read_csv(dataset_path, header=0)
    df["fname"] = data_path + df["fname"]

    # Filter for new entries not already in pickle_df
    new_entries = df[~df["fname"].isin(pickle_df["fname"])]

    # Process new entries
    if not new_entries.empty:
        new_entries["cleaned_text"] = new_entries["fname"].progress_apply(pdf_to_text)
        new_entries["tokenized_text"] = new_entries["cleaned_text"].progress_apply(
            clean_and_tokenize
        )

        # Append new entries to the original pickle_df
        updated_pickle_df = pd.concat([pickle_df, new_entries], ignore_index=True)
    else:
        updated_pickle_df = pickle_df  # No new entries to add

    # Save updated DataFrame
    updated_pickle_df.to_pickle(pickle_path)

    return updated_pickle_df


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


# ### Alternate batch processing
def clean_and_tokenize_chunk(chunk):
    """
    Tokenizes a single chunk of text.
    """
    doc = nlp(chunk)
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]


def batch_tokenize_texts(texts, batch_size=1000, chunk_size=1000000):
    """
    Tokenize a list of texts in batches, handling long texts by processing in chunks.

    :param texts: The list of texts to be tokenized.
    :param batch_size: Number of texts to process in a single batch.
    :param chunk_size: Maximum chunk size in characters for each text.
    :return: A list of lists, where each sublist contains the tokens of a text.
    """
    processed_texts = []
    for text in texts:
        # If the text is longer than chunk_size, split it into chunks
        if len(text) > chunk_size:
            tokens_all_chunks = []
            for start in range(0, len(text), chunk_size):
                end = start + chunk_size
                chunk = text[start:end]
                # Tokenize the chunk and extend the list of tokens
                tokens_all_chunks.extend(clean_and_tokenize_chunk(chunk))
            processed_texts.append(tokens_all_chunks)
        else:
            # For texts that don't exceed the chunk size, process as usual
            tokens = clean_and_tokenize_chunk(text)
            processed_texts.append(tokens)

    return processed_texts


def main():
    dataset_path = "/uw/invest-data/classify_presentations/data/dataset.csv"
    DATA_PATH = "/dave/presentations/"

    dfpickle_path = "/dave/data/df.pkl"
    force = False
    if not os.path.exists(dfpickle_path) or force:
        print("Warning, is dave mounted?")
    else:
        print("Going to load df from pickle file")

    if os.path.exists(dfpickle_path) and not force:
        df = pd.read_pickle(dfpickle_path)
    else:
        df = pd.read_csv(dataset_path, header=0)
        df["fname"] = DATA_PATH + df["fname"]
        tqdm.pandas(desc="Processing documents")
        df["cleaned_text"] = df["fname"].progress_apply(pdf_to_text)

        df["tokenized_text"] = df["cleaned_text"].progress_apply(clean_and_tokenize)


if __name__ == "__main__":
    main()