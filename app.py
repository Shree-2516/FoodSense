from pathlib import Path
import pickle
import math
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

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

# Perform NLP on Reviews
if "reviews_list" in df.columns:
    df["sentiment"] = df["reviews_list"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    
    def extract_keywords(text):
        try:
            blob = TextBlob(str(text))
            words = [w.lower() for w in blob.words if len(w) > 4]
            from collections import Counter
            top_words = [w for w, c in Counter(words).most_common(3)]
            return ", ".join(top_words)
        except:
            return ""
            
    df["keywords"] = df["reviews_list"].apply(extract_keywords)
else:
    df["sentiment"] = 0.0
    df["keywords"] = ""


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


def _format_preferences(budget, rating, cuisine, food="", location="any"):
    return {
        "budget": f"Up to INR {budget}" if budget != "any" else "Any budget",
        "rating": f"Rating {rating}+"
        if rating != "any"
        else "Any rating",
        "cuisine": cuisine.strip() if cuisine and cuisine.strip() else "Any cuisine",
        "food": food.strip() if food and food.strip() else "Any dish",
        "location": location if location != "any" else "Any area",
    }


def recommend(name, budget="any", rating="any", cuisine="", target_location="any", is_veg=False, has_outdoor=False, online_order=False, book_table=False, top_n=6):
    if name not in df["name"].values:
        return []

    idx = df[df["name"] == name].index[0]
    if target_location != "any":
        selected_location = target_location
    else:
        selected_location = df.iloc[idx]["location"]
    query_vec = tfidf_matrix[idx]
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    hybrid_scores = []
    for i, sim_score in enumerate(similarity):
        row = df.iloc[i]
        try:
            rating_val = float(row["rate"])
        except Exception:
            rating_val = 0.0
            
        try:
            votes_val = int(row["votes"])
        except Exception:
            votes_val = 0
            
        normalized_rating = rating_val / 5.0
        log_votes = math.log10(votes_val + 1)
        final_score = (0.5 * sim_score) + (0.3 * normalized_rating) + (0.2 * log_votes)
        
        if "sentiment" in row and pd.notna(row["sentiment"]):
            final_score += float(row["sentiment"]) * 0.2
            
        hybrid_scores.append((i, final_score))

    distances = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    results = []
    seen_names = set()

    for i, h_score in distances:
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
            
        if restaurant["name"] in seen_names:
            continue

        if budget != "any" and restaurant_cost > int(budget):
            continue

        if rating != "any" and restaurant_rating < float(rating):
            continue

        if cuisine and cuisine.strip() and cuisine.strip().lower() not in str(restaurant["cuisines"]).lower():
            continue

        if is_veg and not restaurant.get("is_veg", False):
            continue
        if has_outdoor and not restaurant.get("has_outdoor", False):
            continue
        if online_order and str(restaurant.get("online_order", "")).lower() != "yes":
            continue
        if book_table and str(restaurant.get("book_table", "")).lower() != "yes":
            continue
            
        seen_names.add(restaurant["name"])

        results.append(
            {
                "name": restaurant["name"],
                "rating": restaurant_rating,
                "cost": restaurant_cost,
                "location": restaurant["location"],
                "cuisine": restaurant["cuisines"],
                "type": restaurant["rest_type"],
                "sentiment": round(restaurant.get("sentiment", 0.0), 2),
                "keywords": restaurant.get("keywords", ""),
                "score": round(h_score, 2),
            }
        )

        if len(results) == top_n:
            break

    return results


def recommend_by_food(food_name, data, min_rating=0, max_cost=None, target_location="any", is_veg=False, has_outdoor=False, online_order=False, book_table=False, top_n=10):
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

    if target_location != "any":
        filtered = filtered[filtered["location"] == target_location]

    if is_veg:
        filtered = filtered[filtered["is_veg"] == True]
    if has_outdoor:
        filtered = filtered[filtered["has_outdoor"] == True]
    if online_order:
        filtered = filtered[filtered["online_order"].str.lower() == "yes"]
    if book_table:
        filtered = filtered[filtered["book_table"].str.lower() == "yes"]

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

    if "sentiment" in filtered.columns:
        filtered["score"] += filtered["sentiment"] * 0.2

    filtered = filtered.sort_values(by="score", ascending=False)
    filtered = filtered.drop_duplicates(subset=["name"])
    filtered = filtered.head(top_n)

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
            "sentiment": round(row.get("sentiment", 0.0), 2),
            "keywords": row.get("keywords", ""),
        }
        for _, row in filtered.iterrows()
    ]


def get_recommendations(user_input, food_input=None, min_rating=0, max_cost=None, cuisine="", target_location="any", is_veg=False, has_outdoor=False, online_order=False, book_table=False):
    if food_input and food_input.strip():
        return recommend_by_food(food_input, df, min_rating, max_cost, target_location, is_veg, has_outdoor, online_order, book_table)

    return recommend(user_input, budget=max_cost if max_cost is not None else "any", rating=min_rating if min_rating else "any", cuisine=cuisine, target_location=target_location, is_veg=is_veg, has_outdoor=has_outdoor, online_order=online_order, book_table=book_table)


@app.route("/")
def home():
    restaurant_list = sorted(df["name"].unique())
    locations = sorted([str(loc) for loc in df["location"].unique() if str(loc).strip() and str(loc).lower() != "nan"])
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
    return render_template("index.html", data=restaurant_list, locations=locations, popular_cuisines=popular_cuisines)


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
    location = request.form.get("location", "any")
    min_rating = float(rating) if rating != "any" else 0
    max_cost = int(budget) if budget != "any" else None

    is_veg = request.form.get("is_veg") == "true"
    has_outdoor = request.form.get("has_outdoor") == "true"
    online_order = request.form.get("online_order") == "true"
    book_table = request.form.get("book_table") == "true"

    if (not restaurant or restaurant.strip() == "") and (not food or food.strip() == ""):
        return render_template(
            "result.html",
            recommendations=[],
            message="Enter a restaurant or a food item to get recommendations.",
            selected=None,
            preferences=_format_preferences("any", "any", "", "", "any"),
            search_mode="empty",
        )

    if food and food.strip():
        result = get_recommendations(restaurant, food, min_rating, max_cost, cuisine, location, is_veg, has_outdoor, online_order, book_table)
        message = None
        if not result:
            message = "No restaurants matched that dish with your selected filters. Try a different food item or relax the filters."

        return render_template(
            "result.html",
            recommendations=result,
            selected=restaurant if restaurant and restaurant.strip() else None,
            message=message,
            preferences=_format_preferences(budget, rating, cuisine, food, location),
            search_mode="food",
        )

    if restaurant not in df["name"].values:
        return render_template(
            "result.html",
            recommendations=[],
            message="Restaurant not found",
            selected=None,
            preferences=_format_preferences(budget, rating, cuisine, food, location),
            search_mode="restaurant",
        )

    result = get_recommendations(restaurant, None, min_rating, max_cost, cuisine, location, is_veg, has_outdoor, online_order, book_table)
    message = None
    if not result:
        message = "No restaurants matched your selected filters. Try changing budget, rating, or cuisine."

    return render_template(
        "result.html",
        recommendations=result,
        selected=restaurant,
        message=message,
        preferences=_format_preferences(budget, rating, cuisine, food, location),
        search_mode="restaurant",
    )


if __name__ == "__main__":
    app.run(debug=True)
