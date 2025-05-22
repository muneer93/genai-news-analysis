from django.db import models
from django.utils import timezone

class VideoAnalysis(models.Model):
    video_title = models.CharField(max_length=300, default="Untitled")
    video_id = models.CharField(max_length=100, default="Unknown ID", unique=True)
    video_url = models.URLField(max_length=500, default="www.youtube.com", unique=True)
    channel_name = models.CharField(max_length=200, default="Unknown Channel", unique=True)
    published_at = models.DateTimeField(default=timezone.now)
    view_count = models.PositiveBigIntegerField(default=0)
    caption_text = models.TextField(default="")
    sentiment_label = models.CharField(max_length=50, default="NEUTRAL")
    sentiment_score = models.FloatField(default=0.0)
    bias_left = models.FloatField(default=0.0)
    bias_center = models.FloatField(default=0.0)
    bias_right = models.FloatField(default=0.0)
    bias_biased = models.FloatField(default=0.0)
    bias_neutral = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.channel_name}, {self.video_title}, {self.video_id}"
