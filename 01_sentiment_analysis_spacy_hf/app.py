import streamlit as st
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline
import subprocess

def load_hf_model():
    model_name = 'sentiment-analysis'
    return pipeline(model_name)

def analyze_sentiment_hf(text):
    # Initialize the sentiment analysis pipeline
    sentiment_pipeline = load_hf_model()

    # Analyze the text
    result = sentiment_pipeline(text)[0]
    
    # Extract sentiment and confidence
    sentiment = result['label']
    confidence = result['score']
    
    return sentiment, confidence

def download_spacy_model():
    model_name = "en_core_web_sm"
    subprocess.run(["python", "-m", "spacy", "download", model_name])

def analyze_sentiment_spacy(text):
    # Load the English language model and add the TextBlob component
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")

    # Process the text
    doc = nlp(text)

    # Get the polarity score
    polarity = doc._.blob.polarity

    # Determine sentiment based on polarity
    if polarity > 0.1:
        sentiment = "POSITIVE"
    elif polarity < -0.1:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return sentiment, polarity

# Set background color based on sentiment
def display_sentiment_with_color(sentiment, source):
    sentiment = sentiment.upper()

    if sentiment == "POSITIVE":
        sentiment_color = "green"
    elif sentiment == "NEGATIVE":
        sentiment_color = "red"
    else:
        sentiment_color = "white"  # Default color for NEUTRAL

    # Display sentiment with background color
    st.markdown(
        f"<p style='background-color: {sentiment_color}; padding: 10px; border-radius: 5px;'>"
        f"{source} sentiment: {sentiment}"
        "</p>",
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Sentiment Analysis")
    
    st.title("Welcome to the Sentiment Analysis app!")
    st.write("This app performs sentiment analysis on text using two different methods: HuggingFace and spaCy.")
    st.write("The sentiment analysis classifies text into three categories: positive, negative, or neutral.")
    st.write("Enter your text in the respective sections below to see how these different methods analyze sentiment.")

    # Section 1: Sentiment Analysis using HuggingFace
    st.header("Sentiment Analysis using HuggingFace")
    huggingface_text = st.text_area("Enter text for HuggingFace analysis:", height=150)

    if st.button("Analyze with HuggingFace"):
        if huggingface_text:
            huggingface_sentiment, huggingface_confidence = analyze_sentiment_hf(huggingface_text)

            display_sentiment_with_color(huggingface_sentiment, "HuggingFace")
            st.write(f"HuggingFace confidence: {huggingface_confidence}")

        else:
            st.warning("Please enter some text for HuggingFace analysis.")
    
    # Section 2: Sentiment Analysis using spaCy
    st.header("Sentiment Analysis using spaCy")
    spacy_text = st.text_area("Enter text for spaCy analysis:", height=150)

    if st.button("Analyze with spaCy"):
        if spacy_text:
            spacy_sentiment, spacy_polarity = analyze_sentiment_spacy(spacy_text)

            display_sentiment_with_color(spacy_sentiment, "spaCy")
            st.write(f"spaCy polarity: {spacy_polarity}")
        else:
            st.warning("Please enter some text for spaCy analysis.")

if __name__ == "__main__":
    main()
