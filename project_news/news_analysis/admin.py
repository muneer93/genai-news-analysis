from django.contrib import admin
from django.utils.html import format_html
from import_export.admin import ExportMixin
from .models import VideoAnalysis


@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(ExportMixin, admin.ModelAdmin):
    list_display = (
        'channel_name',
        'video_title',
        'bias_colored_bar',
        'published_at',
    )
    list_filter = ('channel_name', 'published_at')
    search_fields = ('video_title', 'channel_name')
    ordering = ('-published_at',)
    readonly_fields = (
        'channel_name',
        'video_title',
        'video_id',
        'video_url',
        'published_at',
        'view_count',
        'sentiment_label',
        'sentiment_score',
        'bias_left',
        'bias_center',
        'bias_right',
        'bias_biased',
        'bias_neutral',
        'caption_text',
    )

    fieldsets = (
        ('Video Info', {
            'fields': (
                'channel_name', 'video_title', 'video_id', 'video_url', 'published_at', 'view_count',
            )
        }),
        ('Sentiment', {
            'fields': ('sentiment_label', 'sentiment_score')
        }),
        ('Bias Scores', {
            'fields': ('bias_left', 'bias_center', 'bias_right', 'bias_biased', 'bias_neutral')
        }),
        ('Transcript', {
            'classes': ('collapse',),
            'fields': ('caption_text',)
        }),
    )

    def bias_colored_bar(self, obj):
        """
        Displays a colored bar chart summarizing left/center/right bias.
        """
        total = obj.bias_left + obj.bias_center + obj.bias_right or 1  # Avoid divide-by-zero
        left_pct = int((obj.bias_left / total) * 100)
        center_pct = int((obj.bias_center / total) * 100)
        right_pct = 100 - left_pct - center_pct

        return format_html(f"""
            <div style="display: flex; width: 100%; height: 10px; border: 1px solid #ccc;">
                <div style="width: {left_pct}%; background-color: #3b82f6;" title="Left ({left_pct}%)"></div>
                <div style="width: {center_pct}%; background-color: #9ca3af;" title="Center ({center_pct}%)"></div>
                <div style="width: {right_pct}%; background-color: #ef4444;" title="Right ({right_pct}%)"></div>
            </div>
        """)

    bias_colored_bar.short_description = "Bias (L / C / R)"
