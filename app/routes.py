from fastapi import APIRouter
from model.predict import SingletonModelLoader, load_tokenizer_and_config

# Initialize router
router = APIRouter()

# Load the model and tokenizer
model_path = "model/next_word_model.h5"
tokenizer_path = "model/tokenizer.pkl"
config = {"max_sequence_len": 20}

# SingletonModelLoader ensures the model is loaded once
model_loader = SingletonModelLoader(model_path)
tokenizer, max_sequence_len = load_tokenizer_and_config(tokenizer_path, config)

@router.post("/predict/")
def predict_next_word(input_text: str):
    """
    Predict the next word for the given input text.
    """
    try:
        next_word = model_loader.predict(input_text, tokenizer, max_sequence_len)
        return {"status": "success", "next_word": next_word}
    except Exception as e:
        return {"status": "error", "message": str(e)}
