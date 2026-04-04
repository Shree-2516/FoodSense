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


def _parse_rating(value):
    if value is None:
        return 0

    cleaned = ''.join(ch for ch in str(value) if ch.isdigit() or ch == '.')
    try:
        return float(cleaned) if cleaned else 0
    except ValueError:
        return 0


def _parse_votes(value):
    if value is None:
        return 0

    cleaned = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(cleaned) if cleaned else 0


def _available_search_columns(df):
    candidates = ['dish_liked', 'menu_item', 'cuisines', 'name', 'rest_type']
    return [column for column in candidates if column in df.columns]


def _series_or_default(df, column, default):
    if column in df.columns:
        return df[column]
    return [default] * len(df)


def recommend_by_food(food_name, df, min_rating=0, max_cost=None):
    food_name = str(food_name).strip().lower()
    if not food_name:
        return df.head(0)

    filtered = df.copy()
    search_columns = _available_search_columns(filtered)
    if not search_columns:
        return df.head(0)

    for column in search_columns:
        filtered[column] = filtered[column].fillna('').astype(str).str.lower()

    matches = filtered[search_columns[0]].str.contains(food_name, regex=False)
    for column in search_columns[1:]:
        matches = matches | filtered[column].str.contains(food_name, regex=False)

    filtered = filtered[matches].copy()

    if filtered.empty:
        return filtered.head(0)

    filtered['rate'] = [_parse_rating(value) for value in _series_or_default(filtered, 'rate', 0)]
    filtered['votes'] = [_parse_votes(value) for value in _series_or_default(filtered, 'votes', 0)]
    filtered['approx_cost(for two people)'] = [
        _parse_cost(value)
        for value in _series_or_default(filtered, 'approx_cost(for two people)', None)
    ]

    filtered = filtered[filtered['rate'] >= min_rating]

    if max_cost is not None:
        filtered = filtered[
            filtered['approx_cost(for two people)'].fillna(float('inf')) <= max_cost
        ]

    if filtered.empty:
        return filtered.head(0)

    max_votes = filtered['votes'].max()
    if max_votes and max_votes > 0:
        filtered['score'] = (
            filtered['rate'] * 0.6 + (filtered['votes'] / max_votes) * 0.4 * 5
        )
    else:
        filtered['score'] = filtered['rate'] * 0.6

    filtered = filtered.sort_values(by='score', ascending=False)
    return filtered.head(10)


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


def get_recommendations(user_input, food_input=None, min_rating=0, max_cost=None):
    if food_input:
        return recommend_by_food(food_input, data, min_rating, max_cost)

    return recommend(user_input)


if __name__ == "__main__":
    print(recommend("Empire Restaurant"))
