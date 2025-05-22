import os
from dotenv import load_dotenv

# Split the keys by comma
YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS")

print('The key is:', YOUTUBE_API_KEYS)