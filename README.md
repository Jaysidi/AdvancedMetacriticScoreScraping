### 🎮 Metacritic Game Scraper

**A high-performance Python script for scraping video game data from Metacritic.com.  
It leverages asyncio, aiohttp, and concurrent.futures to optimize performance and drastically reduce scraping time.**

### 🚀 Features:

✅ Asynchronous HTTP requests using aiohttp for faster data retrieval  
✅ Parallel execution with asyncio to fetch multiple scores simultaneously  
✅ Advanced concurrency management with ThreadPoolExecutor (max 5 threads for optimized resource usage)  
✅ Game metadata extraction, including ranking, release date, ratings, platforms, developers, publishers, and scores  

### 📌 Requirements:

Make sure you have the required dependencies installed:
```python
pip install aiohttp beautifulsoup4 fake_useragent numpy pandas
```
### 🔧 Usage:

Run the script from the command line:

```python
python OptimizedWebScraping-Metacritic.py
```

By default, the script scrapes 10 pages from Metacritic, but you can adjust this limit in the SCRAPE_TO_PAGE variable.

### 📊 Output:

The scraped data is saved in CSV files.
The script also supports data merging and optional database storage.

### ⚠️ Disclaimer:

This project is for educational purposes only. Please respect Metacritic's terms of service and avoid overloading their servers.