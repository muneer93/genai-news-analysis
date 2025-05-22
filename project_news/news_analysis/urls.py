from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # Homepage/dashboard
    path('analyze/', views.analyze_video, name='analyze_video'),  # Link to Streamlit
    path('channel/<str:channel_name>/', views.channel_detail, name='channel_detail'),  # Per-channel detail page
]
