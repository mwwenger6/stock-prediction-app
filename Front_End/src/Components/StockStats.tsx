import React, { useState, useEffect } from 'react';
import {Container} from 'react-bootstrap';
import GetPriceUpdate from "../Services/GetPriceUpdate";

interface StockGraphProps {
    symbol: string | undefined;
}
interface StockCandle {
    "c": number, //current price
    "h": number, //high price of day
    "l": number, //low price of day
    "o": number, //open price of day
    "pc": number, //previous close price
    "dp": number, //daily percentage change
    "t": number //not sure lol
}
const StockStats = ({ symbol } : StockGraphProps) => {
    const getPrice = GetPriceUpdate;
    const defaultStockCandle: StockCandle = {
        "c": 0,
        "h": 0,
        "l": 0,
        "o": 0,
        "pc": 0,
        "dp": 0,
        "t": 0
    };

    const [stockCandle, setStockCandle] = useState(defaultStockCandle)

    useEffect(() => {

        //Fetch price data on load
        const fetchData = async () => {
            try {
                if(symbol == undefined)
                    console.error('Invalid stock symbol')
                else{
                    const stockData = await getPrice(symbol);
                    console.log(stockData)
                    setStockCandle(stockData)
                }
            }
            catch (error) {
                console.error('Error fetching prices:', error);
            }
        };

        fetchData();
    }, [symbol]);
    return (<>
        <h1 style={{ fontWeight: 300, lineHeight: 1.2 }}> Today's Statistics </h1>
        {/* Set width to 75% on large screens */}
        <hr className="w-75 d-none d-lg-block mx-auto" />
        <Container>
            <div className={"row mb-3"}>
                <div className={"col-2"}> </div>
                <div className={"col-lg-4 col-12"}>
                    <div className={"d-flex justify-content-between"}>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> Current Price: </span>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> {stockCandle.c}</span>
                    </div>
                    <div className={"d-flex justify-content-between"}>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> High Price: </span>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> {stockCandle.h}</span>
                    </div>
                    <div className={"d-flex justify-content-between"}>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> Low Price: </span>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> {stockCandle.l}</span>
                    </div>
                </div>
                <div className={"col-lg-4 col-12"}>
                    <div className={"d-flex justify-content-between"}>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> Prev. Close Price: </span>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> {stockCandle.pc}</span>
                    </div>
                    <div className={"d-flex justify-content-between"}>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> Open Price: </span>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> {stockCandle.c}</span>
                    </div>
                    <div className={"d-flex justify-content-between"}>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> % Change Today: </span>
                        <span className={"display-6"} style={{fontSize: 2 + 'em'}}> {stockCandle.dp.toFixed(2)}%</span>
                    </div>
                </div>
            </div>
        </Container>
    </>)

}

export default StockStats;