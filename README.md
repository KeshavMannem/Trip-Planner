# Trip Planner

Trip Planner is an AI-driven travel assistant that uses **semantic search + RAG** to answer real-time queries about flights, hotels, and destinations.

---

## Features

### Semantic Search + Retrieval
- Uses **SentenceTransformer embeddings** to index travel data.
- Stores and retrieves relevant context using **ChromaDB**.
- Supports similarity search for hotels, destinations, and travel criteria.

### Live Travel Data Scraping
- Scrapes real-time listings from **Booking.com** and **Kayak** using BeautifulSoup.
- Extracts hotel names, prices, ratings, flight details, and location metadata.
- Automatically cleans and structures scraped data for embedding.

### RAG-Powered Travel Suggestions
- Combines retrieved context with **Llama 3** to produce:
  - Personalized hotel recommendations  
  - Destination comparisons  
  - Travel itinerary suggestions  

### Natural Language Query Interface
Users can ask questions like:
- “Find me a hotel in Tokyo under $200 with good reviews.”
- “Compare flights from SFO to NYC next weekend.”
- “Suggest a 3-day itinerary for Barcelona.”

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Scraping | BeautifulSoup, Requests |
| Embeddings | SentenceTransformers |
| Vector Store | ChromaDB |
| LLM | Llama 3 (Ollama) |
| Pipeline | Python, RAG architecture |


