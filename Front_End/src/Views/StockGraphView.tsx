import {Container} from "react-bootstrap";
import StockGraph from "../Components/StockGraph";
import { useParams } from 'react-router-dom';
import GetPriceUpdate from "../Services/GetPriceUpdate";
import {useEffect, useState} from "react";
import StockStats from "../Components/StockStats";

const StockGraphView = () => {
    const params = useParams();
    const symbol = params.symbol;

    return (
        <Container>
            <div className="floatingDiv m-4">
                <StockGraph symbol={ symbol }/>
                <hr/>
                <StockStats symbol={symbol}/>
            </div>
        </Container>
    );
}

export default StockGraphView;