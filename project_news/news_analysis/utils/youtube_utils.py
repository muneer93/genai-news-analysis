from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import re
import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS")

if not YOUTUBE_API_KEYS:
    raise ValueError("Missing YOUTUBE_API_KEYS in environment variables")

API_KEYS_LIST = [key.strip() for key in YOUTUBE_API_KEYS.split(",")]

# Create clients for all keys once, store in a list
YOUTUBE_CLIENTS = [build("youtube", "v3", developerKey=key) for key in API_KEYS_LIST]

# Simple round-robin index stored in Streamlit session state or global
_api_key_index = 0

def get_youtube_client():
    global _api_key_index
    client = YOUTUBE_CLIENTS[_api_key_index]
    _api_key_index = (_api_key_index + 1) % len(YOUTUBE_CLIENTS)
    return client

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"embed/([^?&]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_video_metadata(video_id):
    try:
        youtube = get_youtube_client()
        response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not response.get("items"):
            print(f"No metadata found for video ID: {video_id}")
            return None

        item = response["items"][0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        return {
            "title": snippet.get("title", "N/A"),
            "channel_title": snippet.get("channelTitle", "N/A"),
            "published_at": snippet.get("publishedAt", "N/A"),
            "view_count": stats.get("viewCount", "N/A"),
            "like_count": stats.get("likeCount", "N/A")
        }
    except Exception as e:
        print(f"Error fetching metadata for {video_id}: {e}")
        return None

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        print(f"Transcript error for {video_id}: {e}")
        return None

def fetch_top_comments(video_id, max_results=10):
    comments = []
    try:
        youtube = get_youtube_client()
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    except Exception as e:
        print(f"Comments error for {video_id}: {e}")
    return comments

def fetch_video_data(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Invalid video URL or video ID not found.")
        return None

    metadata = fetch_video_metadata(video_id)
    transcript = fetch_transcript(video_id)
    comments = fetch_top_comments(video_id)

    return {
        "video_id": video_id,
        "metadata": metadata,
        "transcript": transcript,
        "comments": comments
    }