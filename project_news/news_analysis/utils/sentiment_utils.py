from transformers import pipeline, AutoTokenizer
from collections import Counter
import torch

# Disable MPS explicitly
if torch.backends.mps.is_available():
    print("MPS is available but we are disabling it to avoid bugs.")

torch.device("cpu")  # Set device to CPU

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer and pipeline (on CPU)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=tokenizer,
    device=-1  # -1 ensures CPU usage
)

MAX_TOKENS = 512  # max for DistilBERT

def chunk_text_by_tokens(text, max_tokens=MAX_TOKENS):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# news_analysis/utils/sentiment_utils.py

def analyze_sentiment(text, tokenizer=None):
    from transformers import pipeline

    sentiment_pipeline = pipeline("sentiment-analysis")

    if tokenizer:
        # Tokenize and truncate to 512 tokens
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        decoded_text = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
        result = sentiment_pipeline(decoded_text)
    else:
        result = sentiment_pipeline(text[:1000])  # crude truncation fallback if tokenizer not provided

    if isinstance(result, list):
        score = result[0]['score']
        label = result[0]['label']
    else:
        score = result['score']
        label = result['label']

    return {
        "label": label,
        "average_score": score
    }

