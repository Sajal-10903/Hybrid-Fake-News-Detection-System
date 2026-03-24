import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(words)

print("Loading dataset...")
df = pd.read_csv("data/train.csv")
print("Dataset columns:", df.columns)

X = df["title"].apply(preprocess)

y = df["real"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 

model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("Model trained successfully")

pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

pickle.dump(y_test, open("y_test.pkl", "wb"))
pickle.dump(X_test, open("X_test.pkl", "wb"))

pickle.dump(X_test_vec, open("x_test_vect.pkl", "wb"))

print("Model, Vectorizer, and Test Data saved successfully!")