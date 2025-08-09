import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
with open("LR_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("file_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.set_page_config(page_title="Fake News Detector",page_icon="📰", layout="centered")
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)

# Fungsi deteksi metadata otomatis
def detect_country(text):
    text = text.lower()
    if any(loc in text for loc in ["jakarta", "indonesia"]):
        return "Indonesia"
    elif any(loc in text for loc in ["white house", "biden", "trump", "usa"]):
        return "USA"
    elif any(loc in text for loc in ["london", "britain", "uk"]):
        return "UK"
    elif any(loc in text for loc in ["delhi", "india", "modi"]):
        return "India"
    else:
        return "Other"

def detect_category(text):
    text = text.lower()
    if any(kw in text for kw in ["president", "government", "election", "policy"]):
        return "Politics"
    elif any(kw in text for kw in ["vaccine", "covid", "doctor", "hospital", "virus"]):
        return "Health"
    elif any(kw in text for kw in ["startup", "ai", "software", "technology", "robot"]):
        return "Technology"
    elif any(kw in text for kw in ["movie", "actor", "singer", "tv", "music"]):
        return "Entertainment"
    else:
        return "Other"

# Input pilihan pengguna
option = st.radio("Pilih metode input:", ["Ketik Teks Manual", "Pilih dari Daftar Berita"])

if option == "Ketik Teks Manual":
    text_input = st.text_area("Tulis berita yang ingin diprediksi:")
else:
    sample_news = [
        "A popular Facebook post claims that drinking lemon water cures cancer.",
        "New Senate majority leader’s main goal for GOP",
        "NASA announces new discoveries from the James Webb Space Telescope.",
        "The President signed a bill aiming to reduce student loan debt across the country.",
        "Trump takes on Cruz, but lightly."
    ]
    text_input = st.selectbox("Pilih salah satu berita:", sample_news)
if st.button("🔍 Prediksi"):
    if text_input:
        text_tfidf = tfidf.transform([text_input])
        pred = model.predict(text_tfidf)[0]

        # Deteksi metadata otomatis
        country = detect_country(text_input)
        category = detect_category(text_input)

        # Tampilkan hasil prediksi
        if pred == "FAKE":
            color = "#ffc0cb"  # pastel pink
            label = "🚨 Berita ini terdeteksi sebagai: <b>FAKE</b>"
        else:
            color = "#ccffcc"  # mint green
            label = "✅ Berita ini terdeteksi sebagai: <b>REAL</b>"

        st.markdown(f"""
            <div style='background-color:{color};padding:15px;border-radius:10px'>
            <h3 style='text-align:center;'>{label}</h3></div>
        """, unsafe_allow_html=True)

        # Tampilkan metadata
        st.markdown("### 🧾 Informasi Tambahan (Deteksi Otomatis)")
        st.markdown(
            f"""
            <p>📍 <b>Negara Asal:</b> <code>{country}</code><br>
            📚 <b>Kategori:</b> <code>{category}</code></p>
            """, unsafe_allow_html=True)
    else:
        st.warning("Silakan masukkan atau pilih teks berita terlebih dahulu.")
#  footer
footer = """
<hr style="margin-top:30px; border-top: 1px solid #bbb;">
<div style="text-align:center; color:#888888; font-size:14px;">
    Developed by <strong>Dsloven Group 7</strong> – DS Batch 49
</div>
"""
st.markdown(footer, unsafe_allow_html=True)


