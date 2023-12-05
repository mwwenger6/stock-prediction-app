
import React, { useEffect } from 'react';

interface StockItem {
    ticker: string;
    price: number;
}
function Ticker () {
const stocks: StockItem[] = [
    {ticker: 'AAPL', price: 150.25 }, //stocks copied from mikes
    {ticker: 'GOOGL', price: 2700.50 },
    { ticker: 'AMZN', price: 3550.75 },
    { ticker: 'MSFT', price: 340.90 },
    { ticker: 'FB', price: 330.40 },
    { ticker: 'TSLA', price: 950.15 },
    { ticker: 'NFLX', price: 580.60 },
    // ... more stocks
];

function createStockElement(stock: StockItem): HTMLElement {
    const element = document.createElement('div');
    element.className = 'stock-item';
    element.textContent = `${stock.ticker}: $${stock.price}`;
    return element;
}

useEffect(() => {
    const ticker = document.getElementById('stock-ticker');
    if (ticker) {
        const scrollingContainer = document.createElement('div');
        scrollingContainer.className = 'scrolling-container';

        for (let i = 0; i < 50; i++) {
            stocks.forEach(stock => {
                scrollingContainer.appendChild(createStockElement(stock));
            });
        }

        ticker.appendChild(scrollingContainer);
    }
}, []); // Use useEffect to handle side effects


    return (
        <div id="stock-ticker" className="ticker-container"></div> // Ensure a JSX element is returned
    );
}

export default Ticker;