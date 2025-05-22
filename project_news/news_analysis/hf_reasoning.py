import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

def query_flant5(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7},
        "options": {"wait_for_model": True},
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    try:
        return response.json()[0]['generated_text']
    except (KeyError, IndexError):
        return "Error: No output from model."
