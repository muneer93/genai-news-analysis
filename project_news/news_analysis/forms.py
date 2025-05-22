# forms.py
from django import forms

class YouTubeURLForm(forms.Form):
    url = forms.URLField(label='YouTube Video URL', required=True)
