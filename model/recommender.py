import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
df = pd.read_csv('../data/zomato.csv')

# Select important columns
df = df[['name', 'cuisines', 'rate', 'approx_cost(for two people)', 'location', 'rest_type', 'dish_liked', 'reviews_list', 'votes', 'online_order', 'book_table']]
df.dropna(subset=['name', 'cuisines', 'location'], inplace=True)

# Clean rating (remove "/5")
df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False)

# Convert cost to string
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)

# Smart Filters
df['is_veg'] = df['cuisines'].str.contains('veg', case=False, na=False) | df['dish_liked'].str.contains('veg', case=False, na=False)
df['has_outdoor'] = df['reviews_list'].str.contains('outdoor', case=False, na=False)

# Combine features (IMPROVED TAGS ✅)
df['features'] = (
    df['cuisines'].fillna('') + ' ' +
    df['dish_liked'].fillna('') + ' ' +
    df['rest_type'].fillna('')
)

# Optional: reduce size if dataset is too large
df = df.head(5000)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# ❌ REMOVE similarity matrix (causes MemoryError)
# similarity = cosine_similarity(tfidf_matrix)

# ✅ Save only required files
pickle.dump(df, open('restaurant.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))

print("✅ Improved model built successfully (optimized & memory safe)")