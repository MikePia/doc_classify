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
logging.basicConfig(filename='document_processing_errors.log', level=logging.INFO)

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


# %% [markdown]
# ### Use one or the othere here.

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
            tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
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


# %%
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
import re

def check_keywords_alternate(text, keyword_list):
    text = text.lower()
    # Create a pattern that matches whole words only, for all keywords
    pattern = r'\b(' + '|'.join([re.escape(keyword) for keyword in keyword_list]) + r')\b'
    return int(bool(re.search(pattern, text)))

def safety_not_run_thing():
    # Convert keywords to lowercase for case-insensitive matching
    keywords = {category: [keyword.lower() for keyword in keyword_list] for category, keyword_list in keywords.items()}

    # Assuming 'tokenized_text' contains space-separated tokens, it should work well with the modified check_keywords function.
    # Just ensure 'tokenized_text' is a string; if it's a list of tokens, you might need to join them first:
    # df['tokenized_text_str'] = df['tokenized_text'].apply(' '.join)

    for category, keyword_list in keywords.items():
        df[category + '_keyword'] = df['tokenized_text'].apply(check_keywords, args=(keyword_list,))


# %%
def combine_tfidf_keyword(df):
    # Step 2: TF-IDF Calculation
    vectorizer = TfidfVectorizer(max_features=5000)  # Adjust number of features as needed
    tfidf_matrix = vectorizer.fit_transform(df['tokenized_text'])

    # Step 3. Combine keyword and tfidf features into a single matrix    
    # Convert binary keyword matches to a matrix
    keyword_features = df[[col for col in df.columns if '_keyword' in col]].to_numpy()
    # Combine TF-IDF features with keyword binary indicators
    combined_features = np.hstack((tfidf_matrix.toarray(), keyword_features))

    # Now `combined_features` is ready for model training, and should be aligned with your labels.
    return combined_features


# %% [markdown]
# # 4. Training the Classification Model
# 
# ### Next Steps (not-implemented)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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
# # Not using the NN below yet
# 

# %%
def train_nn(X_train, y_train, X_test, y_test):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64))  # Adjust according to the TF-IDF feature size
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))  # Three categories

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
    return model

# %% [markdown]
# # 5. Classification of New Documents --- not ready:
# - **Predicting Categories:** Use the trained model to predict categories for new documents after preprocessing and vectorization.

# %%
def predict_category(text):
    clean_text = clean_and_tokenize(text)
    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)
    return prediction

# %% [markdown]
# # 6. Evaluation and Iteration not ready
# 
# - **Evaluation:** Use metrics like accuracy, precision, recall, and F1 score to evaluate your model on the test set.

# %%
from sklearn.metrics import classification_report
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# %% [markdown]
# - **Iteration:** Based on evaluation results, iterate over your model by tuning hyperparameters, trying different models (e.g., BERT for text classification), or using more advanced text vectorization techniques.

# %% [markdown]
# # 7. Scalability and Optimization
# 
# - Consider parallel processing or distributed computing for preprocessing steps if you face performance bottlenecks.
# - Explore incremental learning or online learning models if retraining on new data frequently.
# 

# %%


# %% [markdown]
# # Implmentation

# %%
dataset_path = '/uw/invest-data/classify_presentations/data/dataset.csv'
DATA_PATH = "/dave/presentations/"
os.path.exists(dataset_path)
df = pd.read_csv(dataset_path, header=0)


# %%

df['fname'] = DATA_PATH + df['fname']


# %%
df['cleaned_text'] = df['fname'].apply(pdf_to_text)



# %%
tqdm.pandas(desc="Processing documents")

df['tokenized_text'] = df['cleaned_text'].progress_apply(clean_and_tokenize)

# %%
keywords = {
    "financial_terms": ['financial', 'investment', 'share price', 'financial metrics', 'investment strategy'],
    "legal_statements": ['confidentiality statement', 'legal disclaimer', 'disclosure statement', 'proprietary information', 'intellectual property'],
    "company_info": ['company overview', 'company analysis', 'business model', 'company performance'],
    "presentation_content": ['visual aids', 'data charts', 'case studies', 'comparative analysis'],
    "company_targets": ['sales targets', 'company targets', 'performance targets'],
    "financial_discussions": ['financial figures', 'financial projections', 'financial results', 'financial language'],
    "regulatory_references": ['SEC filings', 'regulatory filings', 'external entities', 'lawsuits'],
    "detail_descriptions": ['loan details', 'product details', 'research and development', 'financial details'],
    "company_specific": ['company specific', 'industry specific', 'company-specific analysis', 'specific company focus']
    # "Other Clusters" category is omitted since it's broad and without specific keywords
}


# Apply keyword matching
for category, keyword_list in keywords.items():
    df[category + '_keyword'] = df['tokenized_text'].apply(check_keywords, args=(keyword_list,))


# %%
features = combine_tfidf_keyword(df)


# %%
X_train, X_test, y_train, y_test = split_data(features, df['presentation'])

# %%
model = train_logistic_regression(X_train, y_train)

# %%
# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate and print the evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='binary'))
print("Recall:", recall_score(y_test, y_pred, average='binary'))
print("F1 Score:", f1_score(y_test, y_pred, average='binary'))


# %%
y_test.shape

# %%
import pandas as pd

# Assuming you have a DataFrame `df_test` corresponding to your test dataset
# And it includes a column 'doc_id' or similar that uniquely identifies each document
# If you don't have such a DataFrame, you can create it from `X_test` and `y_test`


type(X_test), type(y_test)
# df_test = pd.DataFrame({'doc_id': X_test.index, 'text': X_test, 'label': y_test})
# df_test

# %%

# First, ensure `X_test` retains its index after splitting so you can merge based on index
misclassified_df = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': y_pred
})

# If `X_test` and `y_test` don't automatically align, you might need to reset the index
# misclassified_df = misclassified_df.reset_index()

# Add a column to indicate whether each prediction is correct
misclassified_df['Correctly Classified'] = misclassified_df['True Label'] == misclassified_df['Predicted Label']

# Filter the DataFrame to only include misclassified documents
misclassified_docs = misclassified_df[~misclassified_df['Correctly Classified']]

# Optionally, join with the original DataFrame (df) to include text or other identifying information
# This step requires that `df` and `misclassified_docs` can be aligned by index or a unique identifier
# Example:
# misclassified_docs = misclassified_docs.join(df[['doc_id', 'text']], how='left')

misclassified_docs



