from transformers import pipeline

# Initialize a zero-shot-classification pipeline using a suitable model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_bias(text: str) -> dict:
    """
    Analyze the bias of the given text using zero-shot classification.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary with labels and their corresponding scores.
    """
    # Candidate labels for bias analysis
    candidate_labels = ["left", "right", "center", "neutral", "biased"]

    # Run classification
    result = classifier(text, candidate_labels)

    # Return the results as a dict with label: score
    return dict(zip(result['labels'], result['scores']))
