import React from 'react';
import { Container } from 'react-bootstrap';

interface Stock {
  name: string;
  ticker: string;
  price: number;
  up?: boolean;
}

const FeaturedStock: React.FC<{ stock: Stock }> = ({ stock }) => {

    const spinner = <div className="spinner-border spinner-border-sm" role="status"> <span className="sr-only"></span> </div>
    return (
        <Container className = "featuredStockBg">
            {stock.price === -1 ? spinner :
            <div className="floatingDiv my-1" style={{width: '200px'}}>
                <p>Name: {stock.name}</p>
                <p>Ticker: {stock.ticker}</p>
                <p>Price: {stock.price.toFixed(2)}</p>
            </div>
            }
    </Container>
  );
};

export default FeaturedStock;
