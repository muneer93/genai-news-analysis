import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

def analyze_sentiment(text):
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 50
        }
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        output = response.json()[0]['generated_text']
        return output
    else:
        return f"Error: {response.status_code}"

def get_sentiment_score(text):
    # You might need to fine-tune this function to extract sentiment scores
    # based on the model's output format
    output = analyze_sentiment(text)
    # Implement logic to extract sentiment score from output
    return output

if __name__ == "__main__":
    text = "I love this product!"
    sentiment_output = analyze_sentiment(text)
    print(sentiment_output)