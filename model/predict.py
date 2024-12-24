import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("model/next_word_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer_data = pickle.load(f)

tokenizer = tokenizer_data["tokenizer"]
max_sequence_len = tokenizer_data["max_sequence_len"]

def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

if __name__ == "__main__":
    seed_text = input("Enter seed text: ")
    next_word = predict_next_word(seed_text)
    if next_word:
        print(f"Next word prediction: {next_word}")
    else:
        print("No prediction could be made.")
