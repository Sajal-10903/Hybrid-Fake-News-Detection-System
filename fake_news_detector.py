import streamlit as st
import pickle
import re
import time
import pandas as pd
import altair as alt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from duckduckgo_search import DDGS

st.set_page_config(page_title="AI Fact Checker", page_icon="📰", layout="centered")

port_stem = PorterStemmer()

@st.cache_resource 
def load_assets():
    v_form = pickle.load(open('vectorizer.pkl', 'rb'))
    l_model = pickle.load(open('model.pkl', 'rb'))
    y_t = pickle.load(open('y_test.pkl', 'rb'))
    x_t_v = pickle.load(open('x_test_vect.pkl', 'rb'))
    return v_form, l_model, y_t, x_t_v

vector_form, load_model, y_test, x_test_vect = load_assets()

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower().split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    return ' '.join(con)

def analyze_news(news):
    news_stemmed = stemming(news)
    vectorized = vector_form.transform([news_stemmed])
    prediction = load_model.predict(vectorized)[0]
    probabilities = load_model.predict_proba(vectorized)[0]
    confidence = max(probabilities) * 100
    return prediction, confidence

def search_internet_for_news(query):
    """Searches exclusively for verified news articles online"""
    try:
        ddgs = DDGS()
        results = list(ddgs.news(query, max_results=3))
        return results
    except Exception as e:
        return str(e) 
    
st.title('📰 Hybrid AI Fake News Detector')
st.markdown("This system uses Machine Learning for pattern recognition and Live Web Search for fact verification.")

sentence = st.text_area("Enter News Headline or Claim:", "", height=120)
predict_btt = st.button("🔍 Analyze & Verify", use_container_width=True)

if predict_btt and sentence:
    with st.spinner("Analyzing patterns and scanning Global News..."):
        time.sleep(1) 

        prediction_class, confidence = analyze_news(sentence)

        sources = search_internet_for_news(sentence)
        
        st.markdown("---")
        st.subheader("Analysis Results")

        if isinstance(sources, str):
            st.warning("⚠️ Live Search is currently restricted by your network. Showing ML Model Prediction only.")
            if prediction_class == 1:
                st.success(f'**Looks Reliable** based on text patterns (Confidence: {confidence:.2f}%)')
            else:
                st.error(f'**Fake News Detected** based on text patterns (Confidence: {confidence:.2f}%)')
            st.code(f"Error Details: {sources}")

        elif len(sources) == 0:
            st.error('🚨 **Unreliable / Fabricated Claim**')
            st.write("Our AI scanned the internet, but **no official news sources** match this claim. It is highly likely to be a personally generated statement, rumor, or fake news.")
            st.info(f"*(ML Pattern Confidence was {confidence:.2f}%, but dismissed due to zero factual evidence)*")

        else:
            st.success(f'✅ **Verified! Found Official News Reports**')
            st.info(f"ML Model Linguistic Confidence: {confidence:.2f}%")
            
            st.markdown("### 🌐 Live Internet Sources:")
            for i, source in enumerate(sources):
                title = source.get('title', 'Article')
                body = source.get('body', 'No description available.')
                url = source.get('url', '#')
                publisher = source.get('source', 'News Outlet')
                
                with st.expander(f"Source {i+1}: {title}"):
                    st.write(body)
                    st.markdown(f"**Publisher:** {publisher} | [Read Full Article Here]({url})")


    with st.expander("📊 View AI Model Accuracy Metrics"):
        y_pred = load_model.predict(x_test_vect)
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2f}%")
        col2.metric("Precision", f"{prec:.2f}%")
        col3.metric("Recall", f"{rec:.2f}%")
        col4.metric("F1 Score", f"{f1:.2f}%")