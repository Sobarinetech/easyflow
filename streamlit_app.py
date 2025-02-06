import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from io import BytesIO
import json
import matplotlib.pyplot as plt
import re

# Load model and tokenizer
@st.cache_resource()
def load_model():
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(prompt, email_content):
    input_text = prompt + email_content[:1000]
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="Escalytics", page_icon="📧", layout="wide")
st.title("⚡Escalytics by EverTech")
st.write("Extract insights, root causes, and actionable steps from emails.")

st.sidebar.header("Settings")
features = {
    "sentiment": st.sidebar.checkbox("Perform Sentiment Analysis"),
    "highlights": st.sidebar.checkbox("Highlight Key Phrases"),
    "response": st.sidebar.checkbox("Generate Suggested Response"),
    "wordcloud": st.sidebar.checkbox("Generate Word Cloud"),
    "grammar_check": st.sidebar.checkbox("Grammar Check"),
    "key_phrases": st.sidebar.checkbox("Extract Key Phrases"),
    "actionable_items": st.sidebar.checkbox("Extract Actionable Items"),
    "root_cause": st.sidebar.checkbox("Root Cause Detection"),
    "culprit_identification": st.sidebar.checkbox("Culprit Identification"),
    "trend_analysis": st.sidebar.checkbox("Trend Analysis"),
    "risk_assessment": st.sidebar.checkbox("Risk Assessment"),
    "severity_detection": st.sidebar.checkbox("Severity Detection"),
    "critical_keywords": st.sidebar.checkbox("Critical Keyword Identification"),
    "export": st.sidebar.checkbox("Export Options"),
}

email_content = st.text_area("Paste your email content here:", height=200)

def extract_key_phrases(text):
    return list(set(re.findall(r"\b[A-Za-z]{4,}\b", text)))

def generate_wordcloud(text):
    word_counts = {}
    for word in text.split():
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def export_pdf(text):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

if email_content and st.button("Generate Insights"):
    try:
        summary = generate_response("Summarize the email:", email_content)
        response = generate_response("Draft a response:", email_content) if features["response"] else ""
        highlights = generate_response("Highlight key points:", email_content) if features["highlights"] else ""

        word_counts = generate_wordcloud(email_content)
        wordcloud_fig = plt.figure(figsize=(10, 5))
        plt.bar(word_counts.keys(), word_counts.values())
        plt.xticks(rotation=45)
        plt.title("Word Frequency")
        plt.tight_layout()
        
        st.subheader("AI Summary")
        st.write(summary)
        if features["response"]:
            st.subheader("Suggested Response")
            st.write(response)
        if features["highlights"]:
            st.subheader("Key Highlights")
            st.write(highlights)
        if features["wordcloud"]:
            st.subheader("Word Cloud")
            st.pyplot(wordcloud_fig)
        if features["export"]:
            export_content = f"Summary:\n{summary}\n\nResponse:\n{response}\n\nHighlights:\n{highlights}\n"
            pdf_buffer = BytesIO(export_pdf(export_content))
            buffer_txt = BytesIO(export_content.encode("utf-8"))
            buffer_json = BytesIO(json.dumps({"summary": summary, "response": response, "highlights": highlights}, indent=4).encode("utf-8"))
            st.download_button("Download as Text", data=buffer_txt, file_name="analysis.txt", mime="text/plain")
            st.download_button("Download as PDF", data=pdf_buffer, file_name="analysis.pdf", mime="application/pdf")
            st.download_button("Download as JSON", data=buffer_json, file_name="analysis.json", mime="application/json")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Paste email content and click 'Generate Insights' to start.")
