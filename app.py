from pathlib import Path
import pickle

from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# Load model artifacts using repo-root paths so the app can be launched from anywhere.
with (MODEL_DIR / "restaurant.pkl").open("rb") as file:
    df = pickle.load(file)

with (MODEL_DIR / "tfidf.pkl").open("rb") as file:
    tfidf = pickle.load(file)

with (MODEL_DIR / "tfidf_matrix.pkl").open("rb") as file:
    tfidf_matrix = pickle.load(file)


def _parse_cost(value):
    if value is None:
        return None

    cleaned = "".join(ch for ch in str(value) if ch.isdigit())
    return int(cleaned) if cleaned else None


def _parse_rating(value):
    if value is None:
        return 0

    cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch == ".")
    try:
        return float(cleaned) if cleaned else 0
    except ValueError:
        return 0


def _parse_votes(value):
    if value is None:
        return 0

    cleaned = "".join(ch for ch in str(value) if ch.isdigit())
    return int(cleaned) if cleaned else 0


def _available_search_columns(data):
    candidates = ["dish_liked", "menu_item", "cuisines", "name", "rest_type"]
    return [column for column in candidates if column in data.columns]


def _series_or_default(data, column, default):
    if column in data.columns:
        return data[column]
    return [default] * len(data)


def _format_preferences(budget, rating, cuisine, food=""):
    return {
        "budget": f"Up to INR {budget}" if budget != "any" else "Any budget",
        "rating": f"Rating {rating}+"
        if rating != "any"
        else "Any rating",
        "cuisine": cuisine.strip() if cuisine and cuisine.strip() else "Any cuisine",
        "food": food.strip() if food and food.strip() else "Any dish",
    }


def recommend(name, budget="any", rating="any", cuisine="", top_n=6):
    if name not in df["name"].values:
        return []

    idx = df[df["name"] == name].index[0]
    selected_location = df.iloc[idx]["location"]
    query_vec = tfidf_matrix[idx]
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    distances = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)

    results = []

    for i, _ in distances:
        restaurant = df.iloc[i]

        try:
            restaurant_rating = float(restaurant["rate"])
        except Exception:
            restaurant_rating = 0

        restaurant_cost = _parse_cost(restaurant["approx_cost(for two people)"])
        if restaurant_cost is None:
            restaurant_cost = 9999

        if restaurant["location"] != selected_location:
            continue

        if restaurant["name"] == name:
            continue

        if budget != "any" and restaurant_cost > int(budget):
            continue

        if rating != "any" and restaurant_rating < float(rating):
            continue

        if cuisine and cuisine.strip() and cuisine.strip().lower() not in str(restaurant["cuisines"]).lower():
            continue

        results.append(
            {
                "name": restaurant["name"],
                "rating": restaurant_rating,
                "cost": restaurant_cost,
                "location": restaurant["location"],
                "cuisine": restaurant["cuisines"],
                "type": restaurant["rest_type"],
            }
        )

        if len(results) == top_n:
            break

    return results


def recommend_by_food(food_name, data, min_rating=0, max_cost=None, top_n=10):
    food_name = str(food_name).strip().lower()
    if not food_name:
        return []

    filtered = data.copy()
    search_columns = _available_search_columns(filtered)
    if not search_columns:
        return []

    for column in search_columns:
        filtered[column] = filtered[column].fillna("").astype(str).str.lower()

    matches = filtered[search_columns[0]].str.contains(food_name, regex=False)
    for column in search_columns[1:]:
        matches = matches | filtered[column].str.contains(food_name, regex=False)

    filtered = filtered[matches].copy()

    if filtered.empty:
        return []

    filtered["rate"] = [_parse_rating(value) for value in _series_or_default(filtered, "rate", 0)]
    filtered["votes"] = [_parse_votes(value) for value in _series_or_default(filtered, "votes", 0)]
    filtered["approx_cost(for two people)"] = [
        _parse_cost(value)
        for value in _series_or_default(filtered, "approx_cost(for two people)", None)
    ]

    filtered = filtered[filtered["rate"] >= min_rating]

    if max_cost is not None:
        filtered = filtered[
            filtered["approx_cost(for two people)"].fillna(float("inf")) <= max_cost
        ]

    if filtered.empty:
        return []

    max_votes = filtered["votes"].max()
    if max_votes and max_votes > 0:
        filtered["score"] = (
            filtered["rate"] * 0.6 + (filtered["votes"] / max_votes) * 0.4 * 5
        )
    else:
        filtered["score"] = filtered["rate"] * 0.6

    filtered = filtered.sort_values(by="score", ascending=False).head(top_n)

    return [
        {
            "name": row.get("name", "Unknown"),
            "rating": row["rate"],
            "cost": row["approx_cost(for two people)"]
            if row["approx_cost(for two people)"] is not None
            else "Not available",
            "location": row.get("location", "Not available"),
            "cuisine": row.get("cuisines", "Not available"),
            "type": row.get("rest_type", "Restaurant"),
            "score": round(row["score"], 2),
        }
        for _, row in filtered.iterrows()
    ]


def get_recommendations(user_input, food_input=None, min_rating=0, max_cost=None, cuisine=""):
    if food_input and food_input.strip():
        return recommend_by_food(food_input, df, min_rating, max_cost)

    return recommend(user_input, budget=max_cost if max_cost is not None else "any", rating=min_rating if min_rating else "any", cuisine=cuisine)


@app.route("/")
def home():
    restaurant_list = sorted(df["name"].unique())
    popular_cuisines = (
        df["cuisines"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(5)
        .index
        .tolist()
    )
    return render_template("index.html", data=restaurant_list, popular_cuisines=popular_cuisines)


@app.route("/insights")
def insights():
    insight_images = [
        {"title": "Top Restaurants by Count", "filename": "top_restaurants.png"},
        {"title": "Top Rated Restaurants", "filename": "top_rated.png"},
        {"title": "Rating Distribution", "filename": "rating_distribution.png"},
        {"title": "Popular Cuisines", "filename": "cuisine_freq.png"},
    ]
    return render_template("insights.html", insight_images=insight_images)


@app.route("/recommend", methods=["POST"])
def get_recommendation():
    restaurant = request.form["restaurant"]
    food = request.form.get("food", "")
    budget = request.form["budget"]
    rating = request.form["rating"]
    cuisine = request.form["cuisine"]
    min_rating = float(rating) if rating != "any" else 0
    max_cost = int(budget) if budget != "any" else None

    if (not restaurant or restaurant.strip() == "") and (not food or food.strip() == ""):
        return render_template(
            "result.html",
            recommendations=[],
            message="Enter a restaurant or a food item to get recommendations.",
            selected=None,
            preferences=_format_preferences("any", "any", "", ""),
            search_mode="empty",
        )

    if food and food.strip():
        result = get_recommendations(restaurant, food, min_rating, max_cost, cuisine)
        message = None
        if not result:
            message = "No restaurants matched that dish with your selected filters. Try a different food item or relax the filters."

        return render_template(
            "result.html",
            recommendations=result,
            selected=restaurant if restaurant and restaurant.strip() else None,
            message=message,
            preferences=_format_preferences(budget, rating, cuisine, food),
            search_mode="food",
        )

    if restaurant not in df["name"].values:
        return render_template(
            "result.html",
            recommendations=[],
            message="Restaurant not found",
            selected=None,
            preferences=_format_preferences(budget, rating, cuisine, food),
            search_mode="restaurant",
        )

    result = get_recommendations(restaurant, None, min_rating, max_cost, cuisine)
    message = None
    if not result:
        message = "No restaurants matched your selected filters. Try changing budget, rating, or cuisine."

    return render_template(
        "result.html",
        recommendations=result,
        selected=restaurant,
        message=message,
        preferences=_format_preferences(budget, rating, cuisine, food),
        search_mode="restaurant",
    )


if __name__ == "__main__":
    app.run(debug=True)
