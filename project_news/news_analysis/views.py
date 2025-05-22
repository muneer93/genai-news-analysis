from django.shortcuts import render, get_object_or_404
from .models import VideoAnalysis
from django.db.models import Avg

def dashboard(request):
    # Distinct channel names
    channels = (
        VideoAnalysis.objects
        .values('channel_name')
        .distinct()
    )
    return render(request, 'news_analysis/dashboard.html', {'channels': channels})


def channel_detail(request, channel_name):
    videos = VideoAnalysis.objects.filter(channel_name=channel_name)

    bias_scores = videos.aggregate(
        left_avg=Avg('bias_left') * 100,
        right_avg=Avg('bias_right') * 100,
        center_avg=Avg('bias_center') * 100,
        biased_avg=Avg('bias_biased') * 100,
        neutral_avg=Avg('bias_neutral') * 100,
    )

    return render(request, 'news_analysis/channel_detail.html', {
        'channel_name': channel_name,
        'videos': videos,
        'bias_scores': bias_scores,
    })


def analyze_video(request):
    return render(request, 'news_analysis/analyze.html')  # placeholder if you want a form later
