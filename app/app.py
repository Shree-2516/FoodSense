from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
df = pickle.load(open('../model/restaurant.pkl', 'rb'))
tfidf = pickle.load(open('../model/tfidf.pkl', 'rb'))
tfidf_matrix = pickle.load(open('../model/tfidf_matrix.pkl', 'rb'))


def _parse_cost(value):
    if value is None:
        return None

    cleaned = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(cleaned) if cleaned else None


def recommend(name, budget='any', rating='any', cuisine='', top_n=6):
    if name not in df['name'].values:
        return []

    idx = df[df['name'] == name].index[0]
    selected_location = df.iloc[idx]['location']
    query_vec = tfidf_matrix[idx]
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    distances = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)

    results = []

    for i, _ in distances:
        restaurant = df.iloc[i]

        try:
            restaurant_rating = float(restaurant['rate'])
        except:
            restaurant_rating = 0

        restaurant_cost = _parse_cost(restaurant['approx_cost(for two people)'])
        if restaurant_cost is None:
            restaurant_cost = 9999

        if restaurant['location'] != selected_location:
            continue

        if restaurant['name'] == name:
            continue

        if budget != 'any' and restaurant_cost > int(budget):
            continue

        if rating != 'any' and restaurant_rating < float(rating):
            continue

        if cuisine and cuisine.strip() and cuisine.strip().lower() not in str(restaurant['cuisines']).lower():
            continue

        results.append({
            "name": restaurant['name'],
            "rating": restaurant_rating,
            "cost": restaurant_cost,
            "location": restaurant['location'],
            "cuisine": restaurant['cuisines'],
            "type": restaurant['rest_type']
        })

        if len(results) == top_n:
            break

    return results


@app.route('/')
def home():
    restaurant_list = sorted(df['name'].unique())
    popular_cuisines = (
        df['cuisines']
        .dropna()
        .astype(str)
        .str.split(',')
        .explode()
        .str.strip()
        .value_counts()
        .head(5)
        .index
        .tolist()
    )
    return render_template(
        'index.html',
        data=restaurant_list,
        popular_cuisines=popular_cuisines
    )


@app.route('/insights')
def insights():
    insight_images = [
        {
            "title": "Top Restaurants by Count",
            "filename": "top_restaurants.png"
        },
        {
            "title": "Top Rated Restaurants",
            "filename": "top_rated.png"
        },
        {
            "title": "Rating Distribution",
            "filename": "rating_distribution.png"
        },
        {
            "title": "Popular Cuisines",
            "filename": "cuisine_freq.png"
        }
    ]
    return render_template('insights.html', insight_images=insight_images)


@app.route('/recommend', methods=['POST'])
def get_recommendation():
    restaurant = request.form['restaurant']
    budget = request.form['budget']
    rating = request.form['rating']
    cuisine = request.form['cuisine']

    if not restaurant or restaurant.strip() == "":
        return render_template(
            'result.html',
            recommendations=[],
            message="Please enter a valid input",
            selected=None
        )

    if restaurant not in df['name'].values:
        return render_template(
            'result.html',
            recommendations=[],
            message="Restaurant not found",
            selected=None
        )

    result = recommend(restaurant, budget, rating, cuisine)
    message = None
    if not result:
        message = "No restaurants matched your selected filters. Try changing budget, rating, or cuisine."

    return render_template(
        'result.html',
        recommendations=result,
        selected=restaurant,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
