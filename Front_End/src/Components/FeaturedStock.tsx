import React from 'react';
import upArrow from '../Resources/green_up_arrow.png'
import downArrow from '../Resources/red_down_arrow.png'
import { Container } from 'react-bootstrap';
import Spinner from "./Spinner";
import Stock from "../Interfaces/Stock";
import {Link, useNavigate} from "react-router-dom";

const FeaturedStock: React.FC<{ stock: Stock }> = ({ stock }) => {

    const imageSrc = stock.up ? upArrow : downArrow;
    const navigate = useNavigate();

    const handleClick = () => {
        navigate(`/Stock/${stock.ticker}`, {replace: true})
    };

    return (
        <Container className = "featuredStockBg">
            <div className="floatingDiv my-1 hoverable" style={{width: '200px'}} onClick={handleClick}>
                {
                    stock.price === -1 ? (<Spinner size={'small'} height={'120px'}/>) : ( <>
                    <p>Name: {stock.name}</p>
                    <p>Ticker: {stock.ticker}</p>
                    <p>Price: {stock.price.toFixed(2)} <img className="arrow" src={imageSrc}/></p> </>)
                }
            </div>
        </Container>
  );
};

export default FeaturedStock;