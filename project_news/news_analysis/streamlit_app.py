import os
import sys
import django
import streamlit as st
from transformers import AutoTokenizer, pipeline
from urllib.parse import urlparse, parse_qs
import plotly.graph_objects as go
from dotenv import load_dotenv
import torch
torch.device("cpu")

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
from news_analysis.hf_reasoning import query_flant5

# Hugging Face tools
sentiment_pipeline = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS")

def extract_video_id(url):
    """Extracts YouTube video ID from URL."""
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

def plot_bias_gauge(bias_left, bias_center, bias_right):
    bias_left_percent = bias_left * 100
    bias_center_percent = bias_center * 100
    bias_right_percent = bias_right * 100

    bias_score = bias_right_percent - bias_left_percent  # -100 (Left) to +100 (Right)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bias_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Bias Balance"},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -30], 'color': 'red'},
                {'range': [-30, 30], 'color': 'lightgray'},
                {'range': [30, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': bias_score
            }
        }
    ))
    return fig

def generate_model_commentary(sentiment, bias):
    sentiment_label = sentiment.get("label", "").capitalize()
    sentiment_score = sentiment.get("average_score", 0)

    bias_left = round(bias.get("left", 0) * 100, 1)
    bias_center = round(bias.get("center", 0) * 100, 1)
    bias_right = round(bias.get("right", 0) * 100, 1)
    bias_biased = round(bias.get("biased", 0) * 100, 1)
    bias_neutral = round(bias.get("neutral", 0) * 100, 1)

    commentary = f"""
        **Model Interpretation:**

        According to our model, the overall agenda being discussed in the video is presented with a **{sentiment_label.lower()} tone**, 
        and the model predicts this with **{round(sentiment_score * 100, 1)}% confidence**.

        - The text leans **{bias_left}% left**, **{bias_center}% center**, and **{bias_right}% right** ideologically.
        - It shows a **{bias_biased}% likelihood** of containing biased language versus a **{bias_neutral}% likelihood** of being neutral.

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
        st.write("## Navigation")
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
                        "average_score": existing_analysis.sentiment_score,
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
                st.write(bias)

                fig = plot_bias_gauge(bias['left'], bias['center'], bias['right'])
                st.plotly_chart(fig)

                with st.expander(" Model Interpretation & Suggestions"):
                    flan_prompt = f"""
                    The following is a transcript extracted from a YouTube news video.
                    Analyze it and answer the following questions to help determine its tone and bias:
                    1. Is this a news report or an interview?
                    2. If it is an interview, does the interviewer ask neutral, unbiased questions or do they appear to lead the interviewee?
                    3. If it is a news report, does the reporting appear biased or neutral?
                    4. What is the main agenda or key takeaway of this content?
                    5. Does the content present facts accurately and fairly?

                    Transcript:
                    {existing_analysis.caption_text[:3000]}
                    """

                    flan_response = query_flant5(flan_prompt)

                    if "Error" in flan_response:
                        st.warning("Model did not return a valid response. Try again later.")
                    else:
                        st.markdown(f"** FLAN-T5 Summary:**\n\n{flan_response}")

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
                        sentiment = analyze_sentiment(transcript, tokenizer)
                        st.write(sentiment)

                if transcript:
                    st.write("## Caption Text (Preview)")
                    st.write(transcript[:500] + "...")

                    bias = analyze_bias(transcript)
                    st.write("## Bias Score")
                    st.write(bias)

                    fig = plot_bias_gauge(bias['left'], bias['center'], bias['right'])
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
                            sentiment_label=sentiment["label"],
                            sentiment_score=sentiment["average_score"],
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