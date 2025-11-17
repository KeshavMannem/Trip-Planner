import chromadb
from sentence_transformers import SentenceTransformer
import subprocess
import re
import requests
from bs4 import BeautifulSoup

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chromadb_persist")
collection = chroma_client.get_or_create_collection(name="travel_data")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}


def extract_locations_for_flight(question):
    """
    Extracts origin and destination locations from questions like:
    'Show me flights from New York to Paris'
    """
    match = re.search(r"from ([A-Za-z ]+) to ([A-Za-z ]+)", question, re.IGNORECASE)
    if match:
        origin = match.group(1).strip()
        destination = match.group(2).strip()
        return origin, destination
    return None, None


def extract_location(question):
    """
    Extracts a single location for hotel queries.
    """
    match = re.search(r"in ([A-Za-z ]+)", question, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def retrieve_relevant_docs(query, top_k=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def format_docs_for_prompt(docs):
    return "\n".join(docs)


def scrape_hotels(location):
    url = f"https://www.booking.com/searchresults.html?ss={location}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("❌ Could not fetch hotel data.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    hotels = []
    for item in soup.select("div[data-testid='property-card']")[:5]:
        name = item.select_one("div[data-testid='title']").get_text(strip=True)
        price_tag = item.select_one("span[data-testid='price-and-discounted-price']")
        price = price_tag.get_text(strip=True) if price_tag else "N/A"
        rating_tag = item.select_one("div[data-testid='review-score']")
        rating = rating_tag.get_text(strip=True) if rating_tag else "N/A"

        hotels.append({
            "type": "hotel",
            "name": name,
            "location": location,
            "price_per_night": price,
            "rating": rating,
        })
    return hotels


def scrape_flights(origin, destination):
    """
    Simulates scraping flights from Kayak (pseudo-selectors—adjust for real scraping).
    """
    url = f"https://www.kayak.com/flights/{origin}-{destination}/2025-07-10"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("❌ Could not fetch flight data.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    flights = []
    for item in soup.select("div.resultWrapper")[:5]:
        airline = item.select_one(".codeshares-airline-names").get_text(strip=True)
        price = item.select_one(".price-text").get_text(strip=True)
        time = item.select_one(".section-times").get_text(strip=True)

        flights.append({
            "type": "flight",
            "airline": airline,
            "route": f"{origin} to {destination}",
            "date": "2025-07-10",
            "price": price,
            "time": time,
        })
    return flights


def ask_ollama(question, context):
    prompt = (
        f"Using the following travel data, answer the question briefly:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
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
        print("❌ Error calling Ollama:", e.stderr)
        return "Failed to get LLM response."


def main():
    user_question = input("Ask about travel data: ")

    # Check if it's about flights
    origin, destination = extract_locations_for_flight(user_question)
    if origin and destination:
        print(f"✅ Detected flight query: from {origin} to {destination}")

        flights = scrape_flights(origin, destination)
        if not flights:
            print("❌ No flights found for this route.")
            return

        docs = []
        for i, item in enumerate(flights):
            text = (
                f"Flight by {item['airline']} from {item['route']} on {item['date']} "
                f"departing at {item['time']} priced at {item['price']}."
            )
            docs.append(text)

            embedding = embedder.encode(text).tolist()
            collection.add(
                ids=[f"flight_{origin}_{destination}_{i}"],
                documents=[text],
                embeddings=[embedding],
                metadatas=[item],
            )

        context = format_docs_for_prompt(docs)

    else:
        # Otherwise, assume it's about hotels
        location = extract_location(user_question)
        if not location:
            print("❌ Could not detect a location in your question.")
            return
        print(f"✅ Detected hotel location: {location}")

        search_results = retrieve_relevant_docs(user_question)
        docs = search_results["documents"][0] if search_results["documents"] else []

        city_in_results = any(
            location.lower() in (doc or "").lower()
            for doc in docs
        )

        if docs and city_in_results:
            print("\n✅ Found relevant data in Chroma for this location.")
            context = format_docs_for_prompt(docs)
        else:
            if docs:
                print("\n⚠️ Found data in Chroma, but it does not mention this location. Scraping live...")
            else:
                print("\n❌ No relevant data found in Chroma. Scraping live...")

            hotels = scrape_hotels(location)
            if not hotels:
                print("❌ No hotels found for this location.")
                return

            docs = []
            for i, item in enumerate(hotels):
                text = (
                    f"Hotel {item['name']} in {item['location']} costs {item['price_per_night']} "
                    f"per night with rating {item['rating']}."
                )
                docs.append(text)

                embedding = embedder.encode(text).tolist()
                collection.add(
                    ids=[f"hotel_{location}_{i}"],
                    documents=[text],
                    embeddings=[embedding],
                    metadatas=[item],
                )

            context = format_docs_for_prompt(docs)

    print("\nContext to send to LLM:\n")
    print(context)

    answer = ask_ollama(user_question, context)
    print("\nLLM answer:")
    print(answer)


if __name__ == "__main__":
    main()
