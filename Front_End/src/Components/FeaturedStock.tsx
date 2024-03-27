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
            <div className="floatingDiv my-1 hoverable d-flex flex-column justify-content-between align-items-center" style={{width: '200px', minHeight: '120px'}} onClick={handleClick}>
                {
                    stock.price === -1 ? (<Spinner size={'small'} height={'120px'}/>) : ( <>
                    <h5 className={"mb-2"}>{stock.name} ({stock.ticker}) </h5>
                    <h5 className={"mb-2 fw-normal"}>{stock.price.toFixed(2)} <img className="arrow" src={imageSrc} style={{width: '200px'}}/></h5> </>)
                }
            </div>
        </Container>
  );
};

export default FeaturedStock;