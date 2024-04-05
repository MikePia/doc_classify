For your large-scale document classification project, here's a strategy using NLTK, SpaCy, TensorFlow, and additional helpful tools like PDFMiner or PyMuPDF for PDF processing. This strategy assumes you're comfortable with Python and the mentioned libraries.

### 1. Preprocessing PDF Documents

- **PDF to Text Conversion:** Use PDFMiner or PyMuPDF to extract text from your PDF files. Handle layout analysis if needed, especially for complex documents.

```python
# Example with PyMuPDF
import fitz  # PyMuPDF
def pdf_to_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

### 2. Text Cleaning and Normalization

- **Text Cleaning:** Use regular expressions (re module) to remove unnecessary characters, numbers, or symbols.
- **Tokenization and Lemmatization:** Use SpaCy for advanced tokenization and lemmatization.

```python
import spacy
nlp = spacy.load("en_core_web_sm")  # Or a larger model as needed

def clean_and_tokenize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)
```

### 3. Feature Extraction

- **TF-IDF Vectorization:** Use Scikit-learn's TfidfVectorizer to convert the cleaned text documents into a matrix of TF-IDF features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
```

### 4. Training the Classification Model

- **Splitting Data:** Use your 1000 classified documents as training data. Ensure you have a balanced dataset for the three categories.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
```

- **Model Selection and Training:** Given the textual nature of your task, models like CNN or LSTM could perform well. TensorFlow/Keras will be used here.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64))  # Adjust according to the TF-IDF feature size
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))  # Three categories

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
```

### 5. Classification of New Documents

- **Predicting Categories:** Use the trained model to predict categories for new documents after preprocessing and vectorization.

```python
def predict_category(text):
    clean_text = clean_and_tokenize(text)
    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)
    return prediction
```

### 6. Evaluation and Iteration

- **Evaluation:** Use metrics like accuracy, precision, recall, and F1 score to evaluate your model on the test set.

```python
from sklearn.metrics import classification_report
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

- **Iteration:** Based on evaluation results, iterate over your model by tuning hyperparameters, trying different models (e.g., BERT for text classification), or using more advanced text vectorization techniques.

### 7. Scalability and Optimization

- Consider parallel processing or distributed computing for preprocessing steps if you face performance bottlenecks.
- Explore incremental learning or online learning models if retraining on new data frequently.

This strategy outlines a foundational approach. Depending on your specific requirements and the performance of initial models, you might need to adapt and extend this strategy, including exploring more sophisticated NLP models and techniques.