import React from 'react';
import { Container } from 'react-bootstrap';

interface Stock {
  name: string;
  ticker: string;
  price: number;
}

const FeaturedStock: React.FC<{ stock: Stock }> = ({ stock }) => {
  return (
    <Container className = "featuredStockBg">
        <div className="floatingDiv my-1" style={{width: '200px'}}>
            <p>Name: {stock.name}</p>
            <p>Ticker: {stock.ticker}</p>
            <p>Price: ${stock.price.toFixed(2)}</p>
        </div>
    </Container>
  );
};

export default FeaturedStock;
