import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

model_name = "meta-llama/Llama-3.3-70B-Instruct"

input_text = "Hello, how are you?"

headers = {
    "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": input_text,
    "parameters": {
        "max_length": 50
    }
}

response = requests.post(
    f"https://api-inference.huggingface.co/models/{model_name}",
    headers=headers,
    json=payload
)

if response.status_code == 200:
    output = response.json()[0]['generated_text']
    print(output)
else:
    print(f"Error: {response.status_code}")