import os
import sys
import django
import streamlit as st
from transformers import pipeline, AutoTokenizer
from urllib.parse import urlparse, parse_qs
import plotly.graph_objects as go
from dotenv import load_dotenv
import requests
import json

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Add the project root (one level up from the current file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_news.settings')
django.setup()

# Now safe to import Django models and your utilities
from news_analysis.models import VideoAnalysis
from django.utils.dateparse import parse_datetime
from utils.youtube_utils import fetch_video_data
from utils.sentiment_utils import analyze_sentiment
from utils.bias_utils import analyze_bias

model_name = "meta-llama/Llama-3.3-70B-Instruct"

def analyze_sentiment_with_llama(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": f"Sentiment analysis: {text}",
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
        return {"label": output.strip()}
    else:
        return {"error": f"Failed to analyze sentiment: {response.status_code}"}

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    return None

def plot_bias_gauge(bias_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+delta",
        value=bias_score,
        title={'text': "Bias Balance", 'font': {'size': 40}},
        delta={'reference': 0, 'increasing': {'color': "RebeccaPurple"}, 'font': {'size': 50}},
        gauge={
            'axis': {'range': [-100, 100], 'visible': True},
            'bar': {'color': "rgba(0, 0, 0, 0)"},
            'bgcolor': "rgba(0, 0, 0, 0)",
            'borderwidth': 0,
            'steps': [
                {'range': [-100, -50], 'color': "#FF3737"},  # Red
                {'range': [-50, 50], 'color': "#D3BD63"},  # Yellow
                {'range': [50, 100], 'color': "#34C759"}  # Green
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': bias_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font={'color': "#4C59D4", 'family': "Arial"},
        height=400
    )
    return fig

def generate_model_commentary(sentiment, bias):
    sentiment_label = sentiment.get("label", "").capitalize()
    commentary = f"""
        **Model Interpretation:**

        According to our model, the overall agenda being discussed in the video is presented with a **{sentiment_label.lower()} tone**.

        - The text leans **{round(bias["left"] * 100, 2)}% left**, **{round(bias["center"] * 100, 2)}% center**, and **{round(bias["right"] * 100, 2)}% right** ideologically.
        - It shows a **{round(bias["biased"] * 100, 2)}% likelihood** of containing biased language versus a **{round(bias["neutral"] * 100, 2)}% likelihood** of being neutral.

        **Suggestions to Improve Neutrality:**
        - Incorporate perspectives from across the political spectrum.
        - Reduce emotionally charged or polarizing words.
        - Present data or facts from multiple reputable sources.
        - Balance criticism with constructive insights.
        """
    return commentary

def main():
    st.title(" YouTube Video Bias & Sentiment Analyzer")

    with st.sidebar:
        analysis_type = st.selectbox("Select Analysis Type", ["Analyze Video", "Database Search"])

    if analysis_type == "Analyze Video":
        video_url = st.text_input("Enter YouTube Video URL:")

        if st.button("Fetch Video Data"):
            if not video_url:
                st.error("Please enter a YouTube video URL.")
                return

            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Could not extract video ID from URL. Please enter a valid YouTube video URL.")
                return

            try:
                existing_analysis = VideoAnalysis.objects.get(video_id=video_id)
                st.success("Found existing analysis in database. Displaying saved results:")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("## Video Information")
                    st.write(f"**Title:** {existing_analysis.video_title}")
                    st.write(f"**Video ID:** {existing_analysis.video_id}")
                    st.write(f"**Channel:** {existing_analysis.channel_name}")
                    st.write(f"**Published At:** {existing_analysis.published_at}")
                    st.write(f"**Views:** {existing_analysis.view_count}")

                with col2:
                    st.write("## Sentiment Analysis")
                    sentiment = {
                        "label": existing_analysis.sentiment_label,
                    }
                    st.write(sentiment)

                st.write("## Caption Text (Preview)")
                st.write(existing_analysis.caption_text[:500] + "...")

                st.write("## Bias Score")
                bias = {
                    "left": existing_analysis.bias_left,
                    "center": existing_analysis.bias_center,
                    "right": existing_analysis.bias_right,
                    "biased": existing_analysis.bias_biased,
                    "neutral": existing_analysis.bias_neutral,
                }
                st.write(f"**Left:** {round(bias['left'] * 100, 2)}%")
                st.write(f"**Center:** {round(bias['center'] * 100, 2)}%")
                st.write(f"**Right:** {round(bias['right'] * 100, 2)}%")
                st.write(f"**Biased:** {round(bias['biased'] * 100, 2)}%")
                st.write(f"**Neutral:** {round(bias['neutral'] * 100, 2)}%")

                bias_score = round((bias['right'] - bias['left']) * 100, 2)
                fig = plot_bias_gauge(bias_score)
                st.plotly_chart(fig)

                with st.expander(" Model Interpretation & Suggestions"):
                    commentary = generate_model_commentary(sentiment, bias)
                    st.markdown(commentary)

            except VideoAnalysis.DoesNotExist:
                st.info("No existing analysis found. Fetching video data...")

                video_data = fetch_video_data(video_url)

                if not video_data:
                    st.error("Failed to fetch video data. Please check the URL and try again.")
                    return

                metadata = video_data.get("metadata", {})
                transcript = video_data.get("transcript")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("## Video Information")
                    st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                    st.write(f"**Video ID:** {video_id}")
                    st.write(f"**Channel:** {metadata.get('channel_title', 'N/A')}")
                    st.write(f"**Published At:** {metadata.get('published_at', 'N/A')}")
                    st.write(f"**Views:** {metadata.get('view_count', 'N/A')}")

                with col2:
                    if transcript:
                        st.write("## Sentiment Analysis")
                        sentiment = analyze_sentiment_with_llama(transcript)
                        st.write(sentiment)

                if transcript:
                    st.write("## Caption Text (Preview)")
                    st.write(transcript[:500] + "...")

                    bias = analyze_bias(transcript)
                    st.write("## Bias Score")
                    st.write(f"**Left:** {round(bias['left'] * 100, 2)}%")
                    st.write(f"**Center:** {round(bias['center'] * 100, 2)}%")
                    st.write(f"**Right:** {round(bias['right'] * 100, 2)}%")
                    st.write(f"**Biased:** {round(bias['biased'] * 100, 2)}%")
                    st.write(f"**Neutral:** {round(bias['neutral'] * 100, 2)}%")

                    bias_score = round((bias['right'] - bias['left']) * 100, 2)
                    fig = plot_bias_gauge(bias_score)
                    st.plotly_chart(fig)

                    with st.expander(" Model Interpretation & Suggestions"):
                        commentary = generate_model_commentary(sentiment, bias)
                        st.markdown(commentary)

                    try:
                        VideoAnalysis.objects.create(
                            video_title=metadata.get("title", "Untitled"),
                            video_id=video_id,
                            video_url=video_url,
                            channel_name=metadata.get("channel_title", "Unknown Channel"),
                            published_at=parse_datetime(metadata.get("published_at")),
                            view_count=metadata.get("view_count", 0),
                            caption_text=transcript,
                            sentiment_label=sentiment.get("label", ""),
                            bias_left=bias["left"],
                            bias_center=bias["center"],
                            bias_right=bias["right"],
                            bias_biased=bias["biased"],
                            bias_neutral=bias["neutral"],
                        )
                        st.success("Analysis results saved to database.")
                    except Exception as e:
                        st.error(f"Failed to save results to database: {e}")
                else:
                    st.warning("No captions available for this video.")

if __name__ == "__main__":
    main()