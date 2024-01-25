const axios = require('axios');
const mysql = require('mysql2/promise');

const finnhubApiKey = 'cln73dhr01qkjffmt80gcln73dhr01qkjffmt810';
const dbConfig = {
    host: '71.113.172.111',
    user: 'appuser',
    password: 'secure_password',   
    database: 'stock_trading_app'
};

async function updateStockPrices() {
    try {
        // Create a MySQL connection
        const connection = await mysql.createConnection(dbConfig);

        // Define the stocks you're interested in
        const stocks = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'META', 'TSLA', 'NFLX'];

        for (const ticker of stocks) {
            // Fetch the latest stock price from Finnhub
            const response = await axios.get(`https://finnhub.io/api/v1/quote?symbol=${ticker}&token=${finnhubApiKey}`);
            const { c: currentPrice } = response.data;

            // Update the stock_prices table with the new price
            const [results] = await connection.execute(
                'UPDATE stock_prices SET price = ?, price_date = NOW() WHERE stock_id = (SELECT stock_id FROM stocks WHERE ticker = ?)',
                [currentPrice, ticker]
            );

            console.log(`Updated ${results.affectedRows} rows for ticker: ${ticker}`);
        }

        await connection.end();
    } catch (error) {
        console.error('Failed to update stock prices:', error);
    }
}

updateStockPrices();
