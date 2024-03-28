import requests
from bs4 import BeautifulSoup
import json

# URL of the page to scrape
url = 'https://finance.yahoo.com/losers/'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    rows = soup.find_all('tr', class_='simpTblRow', limit=10)
    data = []
    
    for row in rows:
        # Extract the ticker and name directly
        ticker = row.find('a', {'data-test': 'quoteLink'}).text
        name = row.find('td', {'aria-label': 'Name'}).text

        # Convert price and percent change to float, removing any formatting
        price_str = row.find('td', {'aria-label': 'Price (Intraday)'}).find('fin-streamer', {'data-field': 'regularMarketPrice'}).text.replace(',', '')
        percent_change_str = row.find('td', {'aria-label': '% Change'}).find('fin-streamer', {'data-field': 'regularMarketChangePercent'}).text.replace('%', '')

        # Convert strings to float
        try:
            price = float(price_str)
        except ValueError:
            price = 0.0  # Or handle the error as appropriate

        try:
            percent_change = float(percent_change_str)
        except ValueError:
            percent_change = 0.0  # Or handle the error as appropriate
        
        row_data = {
            "Ticker": ticker,
            "Name": name,
            "Price": price,
            "PercentChange": percent_change,
            "IsPositive": percent_change > 0  # Determine positivity based on the value
        }
        data.append(row_data)
    
    json_data = json.dumps(data)
    post_url = 'https://localhost:7212/Discovery/AddVolatileStocks'
    
    post_response = requests.post(post_url, data=json_data, headers={'Content-Type': 'application/json'}, verify=False)
    
    if post_response.status_code == 200:
        print("Data posted successfully.")
    else:
        print("Failed to post the data:", post_response.text)
else:
    print("Failed to retrieve the webpage")
