import chromadb
from sentence_transformers import SentenceTransformer
import subprocess
import re

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chromadb_persist")
collection = chroma_client.get_or_create_collection(name="travel_data")

# Simple keyword extraction with regex
def extract_location(question):
    """
    A very simple heuristic to extract location from the question.
    You can replace this with spaCy or a better NLP library.
    """
    match = re.search(r"in (\w+)", question, re.IGNORECASE)
    if match:
        return match.group(1)
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

def scrape_live_hotels(location="Paris"):
    """
    Scrape Booking.com for demonstration.
    """
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.booking.com/searchresults.html?ss={location}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch live hotel data.")
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
        print("Error calling Ollama:", e.stderr)
        return "Failed to get LLM response."

def main():
    user_question = input("Ask about travel data: ")

    # Extract location
    location = extract_location(user_question)
    if not location:
        print("❌ Could not detect a location in your question.")
        return

    print(f"\n✅ Detected location: {location}")

    # Retrieve from Chroma
    search_results = retrieve_relevant_docs(user_question)
    docs = search_results["documents"][0] if search_results["documents"] else []

    # Check if any docs mention this location
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

        hotels = scrape_live_hotels(location)
        if not hotels:
            print("❌ No hotels found for this location.")
            return

        docs = []
        for i, item in enumerate(hotels):
            text = (
                f"Hotel {item['name']} in {item['location']} costs {item['price_per_night']} per night with rating {item['rating']}."
            )
            docs.append(text)

            # Optionally store in Chroma
            embedding = embedder.encode(text).tolist()
            collection.add(
                ids=[f"live_{location}_{i}"],
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
