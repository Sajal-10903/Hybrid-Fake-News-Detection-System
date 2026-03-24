# 📰 Hybrid AI Fake News Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow)
![DuckDuckGo](https://img.shields.io/badge/DuckDuckGo-Live%20Search-success)

An advanced, real-time Fake News Detection System built as a Hybrid Application. This project combines the power of **Machine Learning (Pattern Recognition)** with **Live Web Search (Fact Verification)** to detect fabricated news and verify authentic claims.

### 📊 Performance Tracking
- **Accuracy:** ~82.74%
- **Precision:** ~83.90%
- **Recall:** ~95.50%
- **F1 Score:** ~89.32%

## ✨ Key Features

* **Hybrid Detection Engine:** Uses a trained Logistic Regression model to analyze text patterns and linguistic cues.
* **Live Web Verification:** Integrates the DuckDuckGo Search API (`DDGS`) to cross-reference claims against live official news sources.
* **Intelligent Fallback System:** Automatically handles API rate limits by falling back to the ML model's linguistic confidence score, explaining the context to the user.
* **Fabricated Claim Detection:** Successfully flags sentences that are grammatically correct but factually non-existent (e.g., rumors or generated text).
* **Interactive UI:** Built with Streamlit for a clean, modern, and user-friendly web interface.

## 🚀 Installation & Setup

Follow these simple steps to set up the project on your local machine:

**1. Clone the Repository**
```bash
git clone <your-github-repo-url>
cd fake-news-detector

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

**2. Create a Virtual Environment (Recommended)**
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

**3. Install Dependencies**
pip install -r requirements.txt

**4. How to Run**
Since the pre-trained model and vectorizer files (.pkl) are already included in the repository, you do not need to retrain the model.

streamlit run fake_news_detector.py
