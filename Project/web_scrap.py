import requests
from bs4 import BeautifulSoup
import re

def scrape_website(url, max_words=500):
    # Fetching the webpage content
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch the webpage")
        return

    # Parsing the webpage content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extracting text from the webpage
    text = soup.get_text()

    # Removing extra whitespaces and newline characters
    text = re.sub(r'\s+', ' ', text)

    # Splitting text into words
    words = text.split()

    # Limiting the words to max_words
    if len(words) > max_words:
        words = words[:max_words]

    # Joining the words back into text
    text = ' '.join(words)

    return text

def save_to_file(text, filename='food_1.txt'):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    url = input("Enter the URL of the website: ")
    scraped_text = scrape_website(url)
    if scraped_text:
        save_to_file(scraped_text)
        print("Data scraped and saved successfully.")
