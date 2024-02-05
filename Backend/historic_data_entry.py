import requests
import pyodbc
import re
import time

# Function to fetch stock data from Twelve Data API
def get_stock_data(symbol, api_key):
    base_url = 'https://api.twelvedata.com/time_series'
    params = {
        'symbol': symbol,
        'interval': '1day',
        'apikey': api_key,
        'outputsize': 1300  # ~5 years of daily close price data
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if(data['status'] == 'error'):
        print(f"Error getting: {symbol} prices")
        return None
    else:
        prices = data['values']
        return prices


def insert_stocks(fname, conn):
    cursor = conn.cursor()
    
    with open(fname, 'r') as file:
        for line in file:
            # Assuming each line in the file contains Ticker and Name separated by space
            arr = re.split(r'\s+', line.strip(), maxsplit=1)
            symbol, name = arr[0], arr[1]

            cursor.execute('INSERT INTO Stocks (StockName, StockSymbol) VALUES (?, ?)', (name, symbol))
            conn.commit()
            
def fetch_stocks(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT StockID, StockSymbol FROM Stocks')
    stocks = cursor.fetchall()
    return stocks

# Function to insert stock data into the SQL Server database
def insert_stock_data(stock_id, stock_data, conn):
    cursor = conn.cursor()

    for entry in stock_data:
        date = entry['datetime']
        close_price = entry['close']

        try:
            sql_query = "INSERT INTO StockData (StockID, StockDate, StockClosePrice) VALUES (?, ?, ?)"
            data = (stock_id, date, close_price)         
            cursor.execute(sql_query, data)
            conn.commit()
            
        except pyodbc.Error as ex:
            print('error inserting data ', ex)
            


if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your Twelve Data API key
    api_key = '446a11fe72f149bd881f0753ad465055'
    
    # Define your SQL Server connection parameters
    server = 'DESKTOP-V8TJUJ7'
    database = 'master'

    # Create a connection string
    connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};'

    # Connect to SQL Server
    try:
        
        conn = pyodbc.connect(connection_string)
        print("Connected to SQL Server successfully!")
        #insert_stocks('S&P500.txt', conn)
        stocks = fetch_stocks(conn)

        for stock in stocks:
            stock_id, stock_symbol = stock

            stock_data = get_stock_data(stock_symbol, api_key)

            if(stock_data != None):
                insert_stock_data(stock_id, stock_data, conn)
                print(stock_symbol, stock_id, 'data inserted')
                
            time.sleep(8)

    except Exception as e:
        print(f"Error connecting to SQL Server: {str(e)}")

    finally:
        # Close the connection when done
        if conn:
            conn.close()