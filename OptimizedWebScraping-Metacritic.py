import aiohttp
import asyncio
import csv
import logging
import numpy as np
import re
import random
import concurrent.futures
from bs4 import BeautifulSoup as bs
from fake_useragent import UserAgent
from Library import merge_csv_files, k_to_thousands

# Logging configuration (Console + File)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("scraping.log", encoding="utf-8"),
        # logging.StreamHandler()
    ]
)


# Definition of the scraped information storage class
class ScrapedGames:
    def __init__(self, elements=None):
        """
        Initializes an instance of ScrapedGames.
        :param elements: A list of elements scraped in the form of dictionaries.
        """
        self.elements = elements if elements is not None else []

    @property
    def element_count(self):
        """
        Returns the number of elements scraped.
        """
        return len(self.elements)

    def add_element(self, element):
        """
        Adds a scraped element to the list.
        :param element: A dictionary containing the element's data.
        """
        if isinstance(element, dict):
            self.elements.append(element)
        else:
            raise ValueError("The element must be a dictionary.")

    def get_elements_by_key(self, key, value):
        """
        Returns a list of elements with a specific value for a given key.
        :param key: The key to search for.
        :param value: The expected value.
        :return: A list of matching elements.
        """
        return [el for el in self.elements if el.get(key) == value]

    def remove_element(self, element):
        """
        Deletes a specific element if it is present.
        :param element: The dictionary representing the element to be deleted.
        """
        if element in self.elements:
            self.elements.remove(element)

    def clear_elements(self):
        """
        Clears the list of scraped items.
        """
        self.elements.clear()

    def export_to_csv(self, file_path):
        """
        Exports scraped items to a CSV file.
        :param file_path: Path of the output CSV file.
        """
        if not self.elements:
            raise ValueError("Nothing to export.")

        keys = self.elements[0].keys()

        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.elements)

    def __repr__(self):
        return f"ScrapedGames(Elements= {self.element_count} elements)"


# Dynamic User-Agent
UA = UserAgent(platforms='desktop')

# Dynamic referer
REFERERS = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://search.yahoo.com/",
    "https://duckduckgo.com/",
]

# URL snippets
game_url_head = 'https://www.metacritic.com'
game_critic_review_url_tail = 'critic-reviews/?platform='
game_user_review_url_tail = 'user-reviews/?platform='
BASE_URL = "https://www.metacritic.com/browse/game/?releaseYearMin=1958&releaseYearMax=2025&page="
# BASE_URL = "https://www.metacritic.com/browse/game/all/all/all-time/new/?releaseYearMin=1958&releaseYearMax=2025&page="
# Define some regex
meta_score_re = re.compile(r"Metascore ([0-9]{1,2}) out of 100")
user_score_re = re.compile(r"User score ([0-9]\.?[0-9]?) out of 10")
reviews_re = re.compile(r"Based on ([0-9]*) Critic Reviews?|(tbd)")

# Maximum number of attempts before giving up
MAX_RETRIES = 3

store_data_lock = asyncio.Lock()
append_data_lock = asyncio.Lock()
counter = 0
file_counter = 0


def get_platforms(game_soup):
    """
    Gather available platforms on game page.
    Args:
      game_soup: bs4.BeautifulSoup, game page soup.
    Returns:
      a list with platforms.
    """
    platforms = list()
    platforms_soup = \
        game_soup.find('div',
                       class_='c-gamePlatformsSection g-grid-container ' \
                              'g-outer-spacing-bottom-medium').find_all('a')
    for platform in platforms_soup:
        platforms.append(platform.find_all('div')[2].text.strip())
    return platforms


async def async_platform_scores(platform, game_url, url_tail, score_re):
    """
    Function to catch game score data for a given platform
    Args:
      platform: e.g. 'PC'
      game_url: url of the game
      url_tail: part of the URL after the game_url
      score_re: regular expression to find the <div> block containing score data
    Return:
        score, positive reviews count, mixed reviews count, negative reviews count
    """
    # Format URL
    review_url = game_url + url_tail + \
                 platform.lower().replace(' ', '-').replace('(', '') \
                     .replace(')', '').replace('/', '')
    # Get the HTML source code and make a soup
    async with aiohttp.ClientSession() as aps_session:
        review_html = await fetch(review_url, aps_session)
    review_soup = bs(review_html, "lxml")

    # Find the <div> block containing score data
    score_div = \
        review_soup.find('div', title=score_re)

    if score_div is not None:
        # Find the score and append it to the list scores
        score = score_div.span.text.strip()

        # Get a list of <div> containing the number of reviews and
        # extract separate values, append them to lists
        reviews_count = review_soup.find('div', class_='c-scoreCount_container u-grid').find_all('span')
        pos_reviews = k_to_thousands(reviews_count[0].text.strip())
        mix_reviews = k_to_thousands(reviews_count[2].text.strip())
        neg_reviews = k_to_thousands(reviews_count[4].text.strip())
    else:
        # Append nan if no <div> block containing score data found
        score = np.nan
        pos_reviews = np.nan
        mix_reviews = np.nan
        neg_reviews = np.nan
    return score, pos_reviews, mix_reviews, neg_reviews


def platform_scores_task(platform, game_url, url_tail, score_re):
    """
    Performs asynchronous scraping in a synchronous thread.
    Calling async_platform_scores
    Args:
      platform: e.g. 'PC'
      game_url: url of the game
      url_tail: part of the URL after the game_url
      score_re: regular expression to find the <div> block containing score data
    Return:
        score, positive reviews count, mixed reviews count, negative reviews count
    """
    return asyncio.run(async_platform_scores(platform, game_url, url_tail, score_re))


def run_platform_scores(platforms, game_url, url_tail, score_re, max_threads=5):
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        platform_scores = dict(zip(platforms,
                                   executor.map(lambda platform:
                                                platform_scores_task(platform=platform, game_url=game_url,
                                                                     url_tail=url_tail, score_re=score_re),
                                                platforms)))
    return platform_scores


async def get_scores(game_url, platforms, score_type):
    """
    Gather critic or user scores, positive reviews count , mixed reviews count
    and negative reviews count for each available platform on game page.
    Args:
      game_url: str, url of the game page.
      platforms: list, list of available platforms.
      score_type: str, 'critic' or 'user'.
    Returns:
      a dict with platforms as keys and tuples
      (critic or user scores, positive reviews count, mixed reviews count,
      negative reviews count) as values.
    """
    # Format url tail depending on targeted score type
    url_tail = game_critic_review_url_tail \
        if score_type == 'critic' \
        else game_user_review_url_tail

    # Set the regex to use depending on targeted score type
    score_re = meta_score_re \
        if score_type == 'critic' \
        else user_score_re
    return run_platform_scores(platforms, game_url, url_tail, score_re, max_threads=5)


async def fetch(url, session, retries=MAX_RETRIES):
    """Performs an HTTP request with retry, random delay and error handling."""
    headers = {
        "User-Agent": UA.random,
        "Referer": random.choice(REFERERS)}

    for attempt in range(1, retries + 1):
        try:
            logging.debug(f"{url} - {headers}")
            # await asyncio.sleep(random.uniform(0.01, 0.1))
            async with session.get(url, headers=headers, timeout=10) as response:
                logging.debug(f"Response status: {response.status}")
                html = await response.text()
                # Ban detection
                if " captcha " in html.lower() or response.status == 403:
                    logging.error(f"🚨 Ban detected")
                    return None
                logging.debug(f"✅ Success (Try {attempt}) - page {url} ")
                return html

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(f"⚠️ Fail {url} (Try {attempt}/{retries}): {e}")
            await asyncio.sleep(2 ** attempt)  # Backoff exponentiel
        except Exception as e:  # Handling other errors
            print(f"⚠️ Unknown error : {e}")
    logging.error(f"❌ Final failure with url {url}")
    return None

async def save_data():
    global page_data
    global file_counter

    file_counter += 1
    csv_file_name = f"ScrapedData_{file_counter:05d}.csv"
    async with store_data_lock:
        page_data.export_to_csv(csv_file_name)
        logging.info(f"💾 Successfully stored {csv_file_name} to disk.")
        page_data.clear_elements()

async def async_scraper(url):
    """Handles asynchronous requests."""
    global page_data
    async with aiohttp.ClientSession() as as_session:
        page_html = await fetch(url, as_session)
    page_soup = bs(page_html, "lxml")
    games = page_soup.find_all('div', class_='c-finderProductCard c-finderProductCard-game')

    run_analyser(games, max_threads=5)  # Concurrent execution

def scraper_task(url):
    """Executes asynchronous scraping in a synchronous thread."""
    asyncio.run(async_scraper(url))


def run_scraper(urls, max_threads=5):
    """Handles concurrent execution of scraping tasks with `concurrent.futures`.."""
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        executor.map(scraper_task, urls)


async def async_analyser(game):
    global counter
    global page_data

    """Analyse html code and manage asynchronous requests."""
    rank_name = game.find('h3', class_='c-finderProductCard_titleHeading').find_all('span')
    rank = np.nan
    if len(rank_name) == 2:
        # Gather game rank:
        rank = rank_name[0].text.strip('. ')
        # Gather game name
        name = rank_name[1].text.strip()
    else:
        # No Rank, gather game name
        name = rank_name[0].text.strip()
    # Gather game release date and rate:
    meta = game.find('div', class_='c-finderProductCard_meta').find_all('span')
    release_date = meta[0].text.strip()
    release_year = release_date[-4:]

    # Gather game rate
    rate = meta[2].text.replace('Rated ', '').strip() if len(meta) > 1 else np.nan

    # Gather game description
    description = game.find('div', class_='c-finderProductCard_description').span.text.strip()

    # Gather game url
    game_url = game_url_head + game.a['href']

    # Cook game page's soup
    async with aiohttp.ClientSession() as session:
        game_page = await fetch(game_url, session)

    game_soup = bs(game_page, "lxml")

    # Gather game platforms
    game_platforms = get_platforms(game_soup)

    # Gather game genre
    gen = game_soup.find('li', class_='c-genreList_item')
    genre = gen.a.text.strip() if gen is not None else np.nan

    # Gather game developer:
    dev = game_soup.find('div', class_='c-gameDetails_Developer u-flexbox u-flexbox-row')
    developer = dev.ul.li.text.strip() if dev is not None else np.nan

    # Gather game publisher:
    pub = game_soup.find('div', class_='c-gameDetails_Distributor u-flexbox u-flexbox-row')
    publisher = pub.find_all('span')[1].text.strip() if pub is not None and len(pub) > 1 else np.nan

    # Gather game meta & game user scores in //
    meta_scores_task = asyncio.create_task(get_scores(game_url, game_platforms, 'critic'))

    # Gather game user scores:
    user_scores_task = asyncio.create_task(get_scores(game_url, game_platforms, 'user'))
    platforms_meta_scores_dict = await meta_scores_task
    platforms_user_scores_dict = await user_scores_task

    # Append gathered values to ScrapedGame object:
    async with append_data_lock:
        counter += 1
        for platform in game_platforms:
            page_data.add_element({
                'Rank': int(rank.replace(',', '')),
                'Name': name,
                'Platform': platform,
                'Developer': developer,
                'Publisher': publisher,
                'Release_date': release_date,
                'Release_year': release_year,
                'Genre': genre,
                'Rate': rate,
                'Description': description,
                'Critic_score': platforms_meta_scores_dict[platform][0],
                'Critic_positive_reviews': platforms_meta_scores_dict[platform][1],
                'Critic_mixed_reviews': platforms_meta_scores_dict[platform][2],
                'Critic_negative_reviews': platforms_meta_scores_dict[platform][3],
                'User_score': platforms_user_scores_dict[platform][0],
                'User_positive_reviews': platforms_user_scores_dict[platform][1],
                'User_mixed_reviews': platforms_user_scores_dict[platform][2],
                'User_negative_reviews': platforms_user_scores_dict[platform][3],
                'Game_url': game_url
            })
        logging.info(f"DONE {counter} : {name}")
        if (SPLIT_VALUE - 5) < page_data.element_count < (SPLIT_VALUE + 5):
            await save_data()


def analyser_task(soup):
    """Executes asynchronous scraping in a synchronous thread."""
    asyncio.run(async_analyser(soup))


def run_analyser(soups, max_threads=5):
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        executor.map(analyser_task, soups)

async def get_max_page():
    async with aiohttp.ClientSession() as mp_session:
        f_page_html = await fetch(BASE_URL + "1", mp_session)
    mp = bs(f_page_html, 'lxml').find_all("span",
                                      class_="c-navigationPagination_itemButtonContent u-flexbox u-flexbox-alignCenter "
                                             "u-flexbox-justifyCenter")[3].text.strip()
    return int(mp.replace(',', ''))

if __name__ == "__main__":
    PAGE_OFFSET = 0
    SCRAPE_TO_PAGE = 10
    MAX_PAGE_ON_WEBSITE = asyncio.run(get_max_page())
    SCRAPE_PAGE_LIMIT = min(MAX_PAGE_ON_WEBSITE, SCRAPE_TO_PAGE)
    SPLIT_VALUE = 2500

    # URLS à scraper
    URLS = [f"{BASE_URL}{i}" for i in range(PAGE_OFFSET+1, SCRAPE_PAGE_LIMIT+1)]

    logging.info(f"📌 Starts scraping - {SCRAPE_PAGE_LIMIT-PAGE_OFFSET} pages to browse. "
                 f"Pages {PAGE_OFFSET+1} to {SCRAPE_PAGE_LIMIT}")

    page_data = ScrapedGames()
    run_scraper(URLS, max_threads=5)  # Exécution concurrente
    if page_data.element_count > 0:
        asyncio.run(save_data())
    merge_csv_files("", "Scraped_Data", "csv", "ScrapedData_?????.csv",
                    save_to_db=True, compress=True)