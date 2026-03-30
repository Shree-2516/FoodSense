import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load saved data
data = pickle.load(open('../model/restaurant.pkl', 'rb'))
tfidf = pickle.load(open('../model/tfidf.pkl', 'rb'))
tfidf_matrix = pickle.load(open('../model/tfidf_matrix.pkl', 'rb'))


def _parse_cost(value):
    if value is None:
        return None

    cleaned = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(cleaned) if cleaned else None


def recommend(name, budget='any', rating='any', cuisine='', top_n=6):
    if name not in data['name'].values:
        return []

    idx = data[data['name'] == name].index[0]
    selected_location = data.iloc[idx]['location']
    query_vec = tfidf_matrix[idx]
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    distances = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)

    results = []

    for i, _ in distances:
        restaurant = data.iloc[i]

        try:
            rest_rating = float(restaurant['rate'])
        except:
            rest_rating = 0

        rest_cost = _parse_cost(restaurant['approx_cost(for two people)'])
        if rest_cost is None:
            rest_cost = 9999

        if restaurant['location'] != selected_location:
            continue

        if restaurant['name'] == name:
            continue

        if budget != 'any' and rest_cost > int(budget):
            continue

        if rating != 'any' and rest_rating < float(rating):
            continue

        if cuisine and cuisine.strip() and cuisine.strip().lower() not in str(restaurant['cuisines']).lower():
            continue

        results.append({
            "name": restaurant['name'],
            "rating": rest_rating,
            "cost": rest_cost,
            "location": restaurant['location'],
            "cuisine": restaurant['cuisines'],
            "type": restaurant['rest_type']
        })

        if len(results) == top_n:
            break

    return results


if __name__ == "__main__":
    print(recommend("Empire Restaurant"))
