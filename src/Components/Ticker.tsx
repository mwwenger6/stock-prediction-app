
import React, { useEffect } from 'react';

interface StockItem {
    ticker: string;
    price: number;
    startingPrice: number;
}

function Ticker () {
const stocks: StockItem[] = [
    {ticker: 'AAPL', price: 150.25, startingPrice:149.40 }, //stocks copied from mikes
    {ticker: 'GOOGL', price: 2700.50, startingPrice:2700.00 },
    { ticker: 'AMZN', price: 3550.75, startingPrice:3555.00 },
    { ticker: 'MSFT', price: 340.90, startingPrice: 340.00 },
    { ticker: 'FB', price: 330.40, startingPrice: 334.00 },
    { ticker: 'TSLA', price: 950.15, startingPrice: 950.00 },
    { ticker: 'NFLX', price: 580.60, startingPrice: 581.00 },
    // ... more stocks
];

function createStockElement(stock: StockItem): HTMLElement {
    const element = document.createElement('div');
    element.className = 'stock-item';
    if(stock.startingPrice > stock.price){
        element.style.color = 'red';
    }
    else if(stock.startingPrice < stock.price){
        element.style.color = 'rgb(41, 200, 41)';
    }
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