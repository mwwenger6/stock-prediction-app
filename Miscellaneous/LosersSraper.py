import requests
from bs4 import BeautifulSoup

# URL of the page to scrape
url = 'https://finance.yahoo.com/losers/'

# Send a GET request to the page
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all <tr> elements with the specified class (the first 10)
    rows = soup.find_all('tr', class_='simpTblRow', limit=10)
    
    for row in rows:
        # For each row, find the data you're interested in. Here's an example for symbol and name:
        symbol = row.find('a', {'data-test': 'quoteLink'}).text
        name = row.find('td', {'aria-label': 'Name'}).text
        price = row.find('td', {'aria-label': 'Price (Intraday)'}).find('fin-streamer', {'data-field': 'regularMarketPrice'}).text
        change = row.find('td', {'aria-label': 'Change'}).find('fin-streamer', {'data-field': 'regularMarketChange'}).text
        percent_change = row.find('td', {'aria-label': '% Change'}).find('fin-streamer', {'data-field': 'regularMarketChangePercent'}).text
        
        print(f"Symbol: {symbol}, Name: {name}, Price (Intraday): {price}, Change: {change}, % Change: {percent_change}")
else:
    print("Failed to retrieve the webpage")
