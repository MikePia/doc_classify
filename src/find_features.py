"""
This script analyzes PDFs and stores features in a database.
It worked great but  it cost about 20 bucks to run over 50 documents of type 
Investor Presentation and Not Investor Presentation.
Extracted about 3000 features.
"""

import os
import fitz  # PyMuPDF
import spacy
from openai import OpenAI
import time

# import json
from dotenv import load_dotenv
import pandas as pd

import json
from features_db.featuresdb import (
    find_document_by_name,
    add_document_with_features,
)


keyfile = os.getenv("HOME") + "/.openai/tokens"
success = load_dotenv(keyfile)
if not success:
    raise ValueError(f"Could not load {keyfile}")
# Initialize OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    chunks = []
    pagecount = 0
    for i, page in enumerate(doc):
        text += page.get_text()
        if len(text.split()) > 2400:
            chunks.append(text)
            text = ""
            pagecount += 1
    if len(text) > 0:
        chunks.append(text)
    return chunks


def extract_text_from_pdf_with_spacy(pdf_path, max_words_per_chunk=1000):
    """
    Extracts text from a given PDF file and chunks it using spaCy, aiming to keep
    chunks under a specified maximum word count to respect token limits.

    Parameters:
    - pdf_path: Path to the PDF file to be processed.
    - max_words_per_chunk: Maximum number of words allowed per chunk.

    Returns:
    - A list of text chunks.
    """
    # Load spaCy's language model
    nlp = spacy.load("en_core_web_sm")

    # Open the PDF file
    doc = fitz.open(pdf_path)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for page in doc:
        # Extract text from the current page
        text = page.get_text()

        # Process the text with spaCy to split into sentences
        spacy_doc = nlp(text)

        for sent in spacy_doc.sents:
            sentence_text = sent.text.strip()
            word_count = len(sentence_text.split())

            # Check if adding the current sentence would exceed the max words per chunk
            if current_word_count + word_count > max_words_per_chunk:
                # If so, join the current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence_text]
                current_word_count = word_count
            else:
                # Otherwise, add the sentence to the current chunk
                current_chunk.append(sentence_text)
                current_word_count += word_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def analyze_document_with_openai(classifier, document_text):
    """Sends document text to OpenAI with a prompt to analyze if it's an 'investor presentation' or not and identify features."""
    assert classifier in [1, 2], "Classifier must be 1 or 2"

    num_chunks = len(document_text)
    system_message = {
            "role": "system",
            "content": "You are a helpful assistant analyzing a document.",
            }   
    
    collected_features = {}

    i = 0
    while i < num_chunks:
        messages = [system_message]
        chunk = document_text[i]
        try:
            messages.append(
                {
                    "role": "user",
                    "content": f"Analyze the following text (part {i+1} of {num_chunks}). Your goal is to find general features that identify this document as {'not ' if classifier == 2 else ''}an investor presentation, features that may help to classify other documents.",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Your response will be in JSON format. The top level key for features will be 'Features Identified'. The feature keys will be meaningful names and a maximum of 2 words long.",
                }
            )
            messages.append({"role": "assistant", "content": f"{chunk}. \n\n"})

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="gpt-4",
                temperature=0.5,
            )
            print(f"Processing chunk {i+1}/{num_chunks}...")
            features_dict = json.loads(chat_completion.choices[0].message.content)
            collected_features.update(features_dict["Features Identified"])
            i += 1  # Move to the next chunk only if successful

        except Exception as e:  # Catch the specific exception for token limit exceeded
            print(f"Error processing chunk {i+1}/{num_chunks}: {e}")
            # Strategy to split the chunk into two and insert back into the list for re-processing
            if len(chunk) > 1:  # Ensure the chunk can be split
                mid_point = len(chunk) // 2
                document_text.insert(i, chunk[:mid_point])
                document_text[i + 1] = chunk[
                    mid_point:
                ]  # Replace current chunk with the latter half
                num_chunks += 1  # Increment the total chunk count
            else:
                print(f"Cannot split chunk {i+1} further. Skipping.")
                i += 1  # Move past this chunk to avoid infinite loop

    return collected_features


def main(pdf_paths, dataset_path, limit):
    """Main function to analyze PDFs and store features."""
    # Load dataset, use the first row as column names
    dataset = pd.read_csv(dataset_path, header=0)
    # dataset = pd.read_csv(dataset_path)

    # Filter for those with presentation == 1 or 2

    dataset = dataset[dataset.presentation.isin([1, 2])].head(50)
    features_by_document = {}
    for index, row in dataset.iterrows():
        if find_document_by_name(row["fname"]) is not None:
            print(f"Skipping {row['fname']}")
            continue
        print(f"Analyzing {row['fname']}")
        filepath = os.path.join(pdf_paths, row["fname"])

        # Extract text from PDF
        document_chunks = extract_text_from_pdf_with_spacy(filepath)

        # Analyze with OpenAI
        classifier = row["presentation"]
        features = analyze_document_with_openai(classifier, document_chunks)
        add_document_with_features(row["fname"], classifier, features)

        # Parse and store features as JSON
        features_by_document[row["fname"]] = features

        #     # Respect rate limits - adjust sleep time as needed
        time.sleep(5)

    # # Save or print the JSON output
    # print(json.dumps(features_by_document, indent=2))


if __name__ == "__main__":
    main("/dave/presentations/", "./data/dataset.csv", 10)
