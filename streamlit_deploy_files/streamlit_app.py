import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Title
st.set_page_config(page_title="DistilBERT Sentiment Analyzer", layout="centered")
st.title("📊 Sentiment Analyzer (DistilBERT)")
st.write("Enter any review or comment to get the predicted sentiment.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    return tokenizer, model

tokenizer, model = load_model()

# Prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class, probs[0].tolist()

# Input
user_input = st.text_area("Enter your text:", height=150)

# On button click
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label_map = {
            0: "😠 Very Negative",
            1: "🙁 Negative",
            2: "😐 Neutral",
            3: "🙂 Positive",
            4: "😄 Very Positive"
        }

        pred_class, prob_list = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {label_map[pred_class]}")
        st.bar_chart(prob_list)
