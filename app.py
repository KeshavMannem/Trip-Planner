from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from httpcore import TimeoutException
import psycopg2
import os
from urllib.parse import urlparse
import subprocess
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import html
from datetime import datetime, timedelta
import csv

# Initialize Flask
app = Flask(__name__)
CORS(app)
trips = []

# Configure database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost")
url = urlparse(DATABASE_URL)
conn = psycopg2.connect("postgresql://localhost")

# Initialize embedding model and Chroma
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chromadb_persist")
collection = chroma_client.get_or_create_collection(name="travel_data")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    )
}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trip/submit", methods=["POST"])
def submit_trip():
    data = request.get_json()
    name = data.get("name")
    destination = data.get("destination")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    budget = data.get("budget")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trip_requests (name, destination, start_date, end_date, budget)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (name, destination, start_date, end_date, budget),
        )
        conn.commit()
        cur.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    

@app.route("/api/trip", methods=["POST"])
def save_trip():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Invalid request"}), 400

    destination = data.get("destination")
    if not destination:
        return jsonify({"success": False, "error": "Destination required"}), 400

    trip = {
        "name": data.get("name"),
        "origin": data.get("origin"),
        "destination": destination,
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "budget": data.get("budget"),
    }
    trips.append(trip)

    hotels = scrape_hotels(destination, trip["start_date"], trip["end_date"])

    response_html = "<h4>Top Hotels:</h4><ul>"
    if hotels:
        for h in hotels:
            url = h.get("url", "#")
            name = html.escape(h.get("name", "Unknown Hotel"))
            location = html.escape(h.get("location", ""))
            price = html.escape(h.get("price_per_night", "N/A"))
            rating = html.escape(h.get("rating", "N/A"))

            link = f"<a href='{url}' target='_blank'>{name}</a>"
            response_html += (
                f"<li>{link} ({location}) | Price: {price} | Rating: {rating}</li>"
            )
    else:
        response_html += "<li>No hotels found.</li>"
    response_html += "</ul>"

    return jsonify({"success": True, "response": response_html})


@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    destination = data.get("destination")
    hotels = scrape_hotels(destination)
    print("SCRAPED HOTELS:", hotels)

    flights = scrape_flights("New York", destination)
    print("SCRAPED FLIGHTS:", flights)

    docs = []

    for hotel in hotels:
        docs.append(
            f"Hotel {hotel['name']} in {hotel['location']} costs {hotel['price_per_night']} "
            f"per night with rating {hotel['rating']}."
        )

    if flights:
        for flight in flights:
            docs.append(
                f"Flight by {flight['airline']} from {flight['route']} on {flight['date']} "
                f"departing at {flight['time']} priced at {flight['price']}."
            )

    if not docs:
        return jsonify({"response": "No hotels or flights were found. Please try again with a different destination."})

    prompt = (
        f"Based on the following travel options to {destination}, suggest the best hotels"
        + (" and flights" if flights else "")
        + ":\n\n"
        + "\n".join(docs)
        + "\n\nAnswer:"
    )
    summary = call_ollama_cli(prompt)

    return jsonify({"response": summary})

def scrape_hotels(destination, start_date=None, end_date=None):
    search_url = f"https://www.booking.com/searchresults.html?ss={destination}"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(search_url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    hotels = []
    cards = soup.select("div[data-testid='property-card']")
    print(f"Found {len(cards)} hotel cards")

    for card in cards[:5]:
        name_tag = card.select_one("div[data-testid='title']")
        name = name_tag.get_text(strip=True) if name_tag else "N/A"

        location_tag = card.select_one("span[data-testid='address']")
        location = location_tag.get_text(strip=True) if location_tag else destination

        # Rating
        rating = "N/A"
        rating_container = card.select_one("div[data-testid='review-score']")
        if rating_container:
            score_div = rating_container.select_one("div:nth-child(1)")
            desc_div = rating_container.select_one("div:nth-child(2)")
            score = score_div.get_text(strip=True) if score_div else ""
            desc = desc_div.get_text(strip=True) if desc_div else ""
            rating = " / ".join(filter(None, [score, desc]))

        # URL with dates
        link_tag = card.select_one("a[data-testid='title-link']")
        if link_tag:
            raw_href = link_tag.get("href", "")
            if raw_href.startswith("http"):
                base_url = raw_href.split("?")[0]
            else:
                base_url = "https://www.booking.com" + raw_href.split("?")[0]
            if start_date and end_date:
                url_with_dates = f"{base_url}?checkin={start_date}&checkout={end_date}"
            else:
                url_with_dates = base_url
        else:
            url_with_dates = ""

        # Load detail page to get price
        price = "N/A"
        if url_with_dates:
            try:
                detail_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                detail_driver.get(url_with_dates)

                # Wait up to 15 seconds for any price element to appear
                wait = WebDriverWait(detail_driver, 15)
                possible_price_selectors = [
                    (By.CSS_SELECTOR, "span[data-testid='price-and-discounted-price']"),
                    (By.CSS_SELECTOR, "span[data-testid='price']"),
                    (By.CSS_SELECTOR, "div[data-testid='price-and-discounted-price']"),
                    (By.CSS_SELECTOR, "div[data-testid='price']"),
                    (By.CSS_SELECTOR, "span[class*='bui-price-display__value']"),
                    (By.CSS_SELECTOR, "div[class*='bui-price-display__value']"),
                ]

                found_price = False
                for selector in possible_price_selectors:
                    try:
                        element = wait.until(EC.presence_of_element_located(selector))
                        price_text = element.text.strip()
                        if price_text:
                            price = price_text
                            found_price = True
                            break
                    except:
                        continue  # Try next selector

                detail_driver.quit()

            except Exception as e:
                print("Error loading detail page:", e)

        hotels.append({
            "type": "hotel",
            "name": name,
            "url": url_with_dates,
            "location": location,
            "price_per_night": price,
            "rating": rating
        })

    driver.quit()
    return hotels


def call_ollama_cli(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return "Failed to get LLM response."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)


# def scrape_flights(origin_city, destination_city, start_date_str):
#     origin_code = get_airport_code(origin_city)
#     destination_code = get_airport_code(destination_city)

#     if not origin_code or not destination_code:
#         print(f"Airport code not found for {origin_city} or {destination_city}")
#         return []

#     url = f"https://www.kayak.com/flights/{origin_code}-{destination_code}/{start_date_str}"

#     options = Options()
#     options.add_argument("--headless=new")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("--window-size=1920,1080")
#     options.add_argument("--start-maximized")
#     options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
#     driver = webdriver.Chrome(options=options)

#     try:
#         driver.get(url)
#         with open("kayak_debug.html", "w", encoding="utf-8") as f:
#             f.write(driver.page_source)

#         # Save screenshot
#         driver.save_screenshot("kayak_debug.png")
#         wait = WebDriverWait(driver, 60)


#         # Handle cookie popup
#         try:
#             wait.until(
#                 EC.element_to_be_clickable(
#                     (By.CSS_SELECTOR, "button[aria-label='Close']")
#                 )
#             ).click()
#             print("✅ Cookie popup dismissed")
#         except TimeoutException:
#             print("⚠️ No cookie popup found")


#         # Wait for flight results container
#         wait.until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, 'div.resultsContainer'))
#         )
#         print("✅ Flight results loaded")

#         flights = []
#         flight_cards = driver.find_elements(By.CSS_SELECTOR, 'div.resultWrapper')[:5]

#         for card in flight_cards:
#             try:
#                 airline = card.find_element(By.CSS_SELECTOR, 'div.airlineName, span.codeshares-airline-names').text.strip()
#             except:
#                 airline = "N/A"

#             try:
#                 time = card.find_element(By.CSS_SELECTOR, 'div.section-times').text.strip()
#             except:
#                 time = "N/A"

#             try:
#                 price = card.find_element(By.CSS_SELECTOR, 'span.price-text').text.strip()
#             except:
#                 price = "N/A"

#             try:
#                 duration = card.find_element(By.CSS_SELECTOR, 'div.duration').text.strip()
#             except:
#                 duration = "N/A"

#             try:
#                 layovers = card.find_element(By.CSS_SELECTOR, 'div.stops-text').text.strip()
#             except:
#                 layovers = "N/A"

#             flights.append({
#                 "type": "flight",
#                 "airline": airline,
#                 "route": f"{origin_code} to {destination_code}",
#                 "date": start_date_str,
#                 "price": price,
#                 "time": time,
#                 "duration": duration,
#                 "layovers": layovers,
#             })

#         return flights

#     finally:
#         driver.quit()

# # Load airport city to IATA code mapping once
# airport_map = {}

# def load_airport_codes(csv_path='airports.dat'):
#     global airport_map
#     with open(csv_path, encoding='utf-8') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             # row format: ID, Name, City, Country, IATA, ICAO, ...
#             city = row[2].strip().lower()
#             iata = row[4].strip()
#             if iata and city:
#                 # If multiple airports per city, this keeps last, consider list for multiple airports
#                 airport_map[city] = iata

# def get_airport_code(city_name):
#     if not airport_map:
#         load_airport_codes()
#     return airport_map.get(city_name.strip().lower(), None)