import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example dataset (replace with your text corpus)
data = """
   I love coding in Python. Python is amazing. Python is easy to learn and powerful.
    I love creating machine learning models."I love programming in Python",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating and powerful",
        "Natural language processing enables cool applications",
        "Artificial intelligence is transforming the world",
        "FastAPI is great for building APIs quickly",
        "Predicting the next word is a fun challenge",
        "Data science combines programming and statistics"
"""

# Preprocess data
def preprocess_data(text, max_words=5000):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    word_index = tokenizer.word_index

    sequences = []
    for sentence in text.split("."):
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max(len(x) for x in sequences)
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding="pre"))

    # Inputs and labels
    x = sequences[:, :-1]
    y = sequences[:, -1]
    vocab_size = len(word_index) + 1  # Actual vocabulary size
    y = np.eye(vocab_size)[y]  # One-hot encode using vocab_size

    return x, y, tokenizer, max_sequence_len

# Create the model
def create_model(vocab_size, max_sequence_len):
    model = Sequential([
        Embedding(vocab_size, 10, input_length=max_sequence_len-1),
        LSTM(100),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # Preprocess data
    x, y, tokenizer, max_sequence_len = preprocess_data(data)

    # Create and train model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size, max_sequence_len)
    model.fit(x, y, epochs=10, verbose=1)

    # Save model and tokenizer
    model.save("model/next_word_model.h5")
    with open("model/tokenizer.pkl", "wb") as f:
        pickle.dump({"tokenizer": tokenizer, "max_sequence_len": max_sequence_len}, f)

    print("Model and tokenizer saved.")