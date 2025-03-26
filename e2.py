import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("scraped_data.csv")

# 1. Handle Duplicates
df = df.drop_duplicates(subset=["unique_id_column"])  # Replace with relevant columns

# 2. Handle Missing Values
# Numerical columns: impute with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical/text columns: impute with mode or placeholder
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna("missing")

# 3. Outlier Detection & Treatment
z_scores = np.abs(stats.zscore(df[num_cols]))
df = df[(z_scores < 3).all(axis=1)]  # Remove outliers beyond 3σ

# 4. Text Preprocessing
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df["text_column"] = df["text_column"].apply(preprocess_text)

# 5. Categorical Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_cats = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())

# 6. Feature Scaling
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_nums, columns=num_cols)

# Final preprocessed dataset
processed_df = pd.concat([scaled_df, encoded_df], axis=1)
# Text Vectorization
tfidf = TfidfVectorizer(max_features=5000)
text_features = tfidf.fit_transform(df["text_column"])

# Alternative: Embeddings (using transformers)
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy()

# (Note: Requires GPU for large datasets)
