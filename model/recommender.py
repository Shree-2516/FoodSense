import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
df = pd.read_csv('../data/zomato.csv')

# Select important columns
df = df[['name', 'cuisines', 'rate', 'approx_cost(for two people)', 'location', 'rest_type']]
df.dropna(inplace=True)

# Clean rating (remove "/5")
df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False)

# Convert cost to string
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)

# Combine features (IMPROVED TAGS ✅)
df['tags'] = (
    df['cuisines'] + " " +
    df['location'] + " " +
    df['rest_type'] + " " +
    "rating_" + df['rate']
)

# Optional: reduce size if dataset is too large
df = df.head(5000)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# ❌ REMOVE similarity matrix (causes MemoryError)
# similarity = cosine_similarity(tfidf_matrix)

# ✅ Save only required files
pickle.dump(df, open('restaurant.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))

print("✅ Improved model built successfully (optimized & memory safe)")