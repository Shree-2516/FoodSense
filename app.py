from pathlib import Path
import math
import pickle
import sqlite3
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
from starlette.middleware.sessions import SessionMiddleware
from textblob import TextBlob
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"
DATABASE_PATH = BASE_DIR / "users.db"
ENV_PATH = BASE_DIR / ".env"

app = FastAPI(title="Restaurant Recommendation System")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


ENV_VALUES = load_env_file(ENV_PATH)
DEFAULT_USER_EMAIL = ENV_VALUES.get("APP_USER_EMAIL", "")
DEFAULT_USER_PASSWORD = ENV_VALUES.get("APP_USER_PASSWORD", "")
SESSION_SECRET = ENV_VALUES.get(
    "FLASK_SECRET_KEY",
    ENV_VALUES.get("FASTAPI_SECRET_KEY", "restaurant-recommender-dev-secret"),
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)


with (MODEL_DIR / "restaurant.pkl").open("rb") as file:
    df = pickle.load(file)

with (MODEL_DIR / "tfidf.pkl").open("rb") as file:
    tfidf = pickle.load(file)

with (MODEL_DIR / "tfidf_matrix.pkl").open("rb") as file:
    tfidf_matrix = pickle.load(file)


if "reviews_list" in df.columns:
    df["sentiment"] = df["reviews_list"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    def extract_keywords(text):
        try:
            blob = TextBlob(str(text))
            words = [word.lower() for word in blob.words if len(word) > 4]
            from collections import Counter

            top_words = [word for word, _count in Counter(words).most_common(3)]
            return ", ".join(top_words)
        except Exception:
            return ""

    df["keywords"] = df["reviews_list"].apply(extract_keywords)
else:
    df["sentiment"] = 0.0
    df["keywords"] = ""


def get_db():
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    if DEFAULT_USER_EMAIL and DEFAULT_USER_PASSWORD:
        cursor.execute("SELECT id FROM users WHERE email = ?", (DEFAULT_USER_EMAIL,))
        existing_user = cursor.fetchone()
        if existing_user is None:
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (DEFAULT_USER_EMAIL, generate_password_hash(DEFAULT_USER_PASSWORD)),
            )

    db.commit()
    db.close()


init_db()


def add_flash(request: Request, message: str):
    flashes = request.session.get("flash_messages", [])
    flashes.append(message)
    request.session["flash_messages"] = flashes


def consume_flash_messages(request: Request) -> list[str]:
    messages = request.session.get("flash_messages", [])
    request.session.pop("flash_messages", None)
    return messages


def current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    db = get_db()
    try:
        return db.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    finally:
        db.close()


def template_response(request: Request, template_name: str, **context):
    base_context = {
        "request": request,
        "current_user": current_user(request),
        "flash_messages": consume_flash_messages(request),
    }
    base_context.update(context)
    return templates.TemplateResponse(request, template_name, base_context)


def redirect_to(route_name: str, request: Request):
    return RedirectResponse(url=request.url_for(route_name), status_code=303)


def require_user(request: Request):
    user = current_user(request)
    if user is None:
        return None, redirect_to("login", request)
    return user, None


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
        "rating": f"Rating {rating}+" if rating != "any" else "Any rating",
        "cuisine": cuisine.strip() if cuisine and cuisine.strip() else "Any cuisine",
        "food": food.strip() if food and food.strip() else "Any dish",
        "location": location if location != "any" else "Any area",
    }


def recommend(
    name,
    budget="any",
    rating="any",
    cuisine="",
    target_location="any",
    is_veg=False,
    has_outdoor=False,
    online_order=False,
    book_table=False,
    top_n=6,
):
    if name not in df["name"].values:
        return []

    idx = df[df["name"] == name].index[0]
    selected_location = target_location if target_location != "any" else df.iloc[idx]["location"]
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

    distances = sorted(hybrid_scores, key=lambda item: item[1], reverse=True)

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


def recommend_by_food(
    food_name,
    data,
    min_rating=0,
    max_cost=None,
    target_location="any",
    is_veg=False,
    has_outdoor=False,
    online_order=False,
    book_table=False,
    top_n=10,
):
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
        filtered["score"] = filtered["rate"] * 0.6 + (filtered["votes"] / max_votes) * 0.4 * 5
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


def get_recommendations(
    user_input,
    food_input=None,
    min_rating=0,
    max_cost=None,
    cuisine="",
    target_location="any",
    is_veg=False,
    has_outdoor=False,
    online_order=False,
    book_table=False,
):
    if food_input and food_input.strip():
        return recommend_by_food(
            food_input,
            df,
            min_rating,
            max_cost,
            target_location,
            is_veg,
            has_outdoor,
            online_order,
            book_table,
        )

    return recommend(
        user_input,
        budget=max_cost if max_cost is not None else "any",
        rating=min_rating if min_rating else "any",
        cuisine=cuisine,
        target_location=target_location,
        is_veg=is_veg,
        has_outdoor=has_outdoor,
        online_order=online_order,
        book_table=book_table,
    )


@app.get("/login", name="login")
async def login_page(request: Request):
    if current_user(request) is not None:
        return redirect_to("home", request)
    return template_response(request, "login.html", email="")


@app.post("/login", name="login_post")
async def login(request: Request):
    if current_user(request) is not None:
        return redirect_to("home", request)

    form = await request.form()
    email = str(form.get("email", "")).strip().lower()
    password = str(form.get("password", ""))

    if not email or not password:
        add_flash(request, "Enter both email and password.")
        return template_response(request, "login.html", email=email)

    db = get_db()
    try:
        user = db.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    finally:
        db.close()

    if user is None or not check_password_hash(user["password_hash"], password):
        add_flash(request, "Invalid email or password.")
        return template_response(request, "login.html", email=email)

    request.session.clear()
    request.session["user_id"] = user["id"]
    return redirect_to("home", request)


@app.get("/register", name="register")
async def register_page(request: Request):
    if current_user(request) is not None:
        return redirect_to("home", request)
    return template_response(request, "register.html", email="")


@app.post("/register", name="register_post")
async def register(request: Request):
    if current_user(request) is not None:
        return redirect_to("home", request)

    form = await request.form()
    email = str(form.get("email", "")).strip().lower()
    password = str(form.get("password", ""))
    confirm_password = str(form.get("confirm_password", ""))

    if not email or not password or not confirm_password:
        add_flash(request, "Fill in all registration fields.")
        return template_response(request, "register.html", email=email)

    if password != confirm_password:
        add_flash(request, "Passwords do not match.")
        return template_response(request, "register.html", email=email)

    db = get_db()
    try:
        existing_user = db.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if existing_user is not None:
            add_flash(request, "That email is already registered. Please log in.")
            return redirect_to("login", request)

        db.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, generate_password_hash(password)),
        )
        db.commit()
    finally:
        db.close()

    add_flash(request, "Registration successful. Please log in.")
    return redirect_to("login", request)


@app.get("/logout", name="logout")
async def logout(request: Request):
    request.session.clear()
    return redirect_to("login", request)


@app.get("/", name="home")
async def home(request: Request):
    _user, redirect_response = require_user(request)
    if redirect_response is not None:
        return redirect_response

    restaurant_list = sorted(df["name"].unique())
    locations = sorted(
        [str(loc) for loc in df["location"].unique() if str(loc).strip() and str(loc).lower() != "nan"]
    )
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
    return template_response(
        request,
        "index.html",
        data=restaurant_list,
        locations=locations,
        popular_cuisines=popular_cuisines,
    )


@app.get("/insights", name="insights")
async def insights(request: Request):
    _user, redirect_response = require_user(request)
    if redirect_response is not None:
        return redirect_response

    insight_images = [
        {"title": "Top Restaurants by Count", "filename": "top_restaurants.png"},
        {"title": "Top Rated Restaurants", "filename": "top_rated.png"},
        {"title": "Rating Distribution", "filename": "rating_distribution.png"},
        {"title": "Popular Cuisines", "filename": "cuisine_freq.png"},
    ]
    return template_response(request, "insights.html", insight_images=insight_images)


@app.post("/recommend", name="recommend")
async def get_recommendation(request: Request):
    _user, redirect_response = require_user(request)
    if redirect_response is not None:
        return redirect_response

    form = await request.form()
    restaurant = str(form.get("restaurant", ""))
    food = str(form.get("food", ""))
    budget = str(form.get("budget", "any"))
    rating = str(form.get("rating", "any"))
    cuisine = str(form.get("cuisine", ""))
    location = str(form.get("location", "any"))
    min_rating = float(rating) if rating != "any" else 0
    max_cost: Optional[int] = int(budget) if budget != "any" else None

    is_veg = form.get("is_veg") == "true"
    has_outdoor = form.get("has_outdoor") == "true"
    online_order = form.get("online_order") == "true"
    book_table = form.get("book_table") == "true"

    if (not restaurant or restaurant.strip() == "") and (not food or food.strip() == ""):
        return template_response(
            request,
            "result.html",
            recommendations=[],
            message="Enter a restaurant or a food item to get recommendations.",
            selected=None,
            preferences=_format_preferences("any", "any", "", "", "any"),
            search_mode="empty",
        )

    if food and food.strip():
        result = get_recommendations(
            restaurant,
            food,
            min_rating,
            max_cost,
            cuisine,
            location,
            is_veg,
            has_outdoor,
            online_order,
            book_table,
        )
        message = None
        if not result:
            message = "No restaurants matched that dish with your selected filters. Try a different food item or relax the filters."

        return template_response(
            request,
            "result.html",
            recommendations=result,
            selected=restaurant if restaurant and restaurant.strip() else None,
            message=message,
            preferences=_format_preferences(budget, rating, cuisine, food, location),
            search_mode="food",
        )

    if restaurant not in df["name"].values:
        return template_response(
            request,
            "result.html",
            recommendations=[],
            message="Restaurant not found",
            selected=None,
            preferences=_format_preferences(budget, rating, cuisine, food, location),
            search_mode="restaurant",
        )

    result = get_recommendations(
        restaurant,
        None,
        min_rating,
        max_cost,
        cuisine,
        location,
        is_veg,
        has_outdoor,
        online_order,
        book_table,
    )
    message = None
    if not result:
        message = "No restaurants matched your selected filters. Try changing budget, rating, or cuisine."

    return template_response(
        request,
        "result.html",
        recommendations=result,
        selected=restaurant,
        message=message,
        preferences=_format_preferences(budget, rating, cuisine, food, location),
        search_mode="restaurant",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
