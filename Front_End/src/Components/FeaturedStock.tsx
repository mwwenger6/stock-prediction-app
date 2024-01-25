import React from 'react';
import upArrow from '../Resources/green_up_arrow.png'
import downArrow from '../Resources/red_down_arrow.png'
import { Container } from 'react-bootstrap';

const API_BASE_URL = 'http://18.116.164.159:3002';

interface Stock {
  name: string;
  ticker: string;
  price: number;
  up?: boolean;
}

const FeaturedStock: React.FC<{ stock: Stock }> = ({ stock }) => {

    const imageSrc = stock.up ? upArrow : downArrow;
    const spinner = <div className="spinner-border spinner-border-sm" role="status"> <span className="sr-only"></span> </div>

    return (
        <Container className = "featuredStockBg">
            {stock.price === -1 ? spinner :
            <div className="floatingDiv my-1" style={{width: '200px'}}>
                <p>Name: {stock.name}</p>
                <p>Ticker: {stock.ticker}</p>
                <p>Price: {stock.price.toFixed(2)} <img className="arrow" src={imageSrc}/></p>
            </div>
            }
    </Container>
  );
};

export default FeaturedStock;
