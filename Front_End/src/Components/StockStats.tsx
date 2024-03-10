import React, { useState, useEffect } from 'react';
import {Container} from 'react-bootstrap';
import GetPriceUpdate from "../Services/GetPriceUpdate";
import DailyData from "../Interfaces/DailyData";

interface StockGraphProps {
    symbol: string | undefined;
}
const StockStats = ({ symbol } : StockGraphProps) => {
    const getPrice = GetPriceUpdate;
    const defaultData : DailyData = {
        "c": 0,
        "h": 0,
        "l": 0,
        "o": 0,
        "pc": 0,
        "dp": 0,
        "t": 0
    };

    const fw :  string = '400';
    const [data, setData] = useState(defaultData)

    useEffect(() => {

        //Fetch price data on load
        const fetchData = async () => {
            try {
                if(symbol == undefined)
                    console.error('Invalid stock symbol')
                else{
                    const stockData = await getPrice(symbol);
                    setData(stockData)
                }
            }
            catch (error) {
                console.error('Error fetching prices:', error);
            }
        };

        fetchData();
    }, [symbol]);
    return (<>
        <h2 style={{ fontWeight: 400, lineHeight: 1.2 }}> Today's Statistics </h2>
        <hr className="mx-auto" />
        <Container>
            <div className={"row"}>
                <div className={"col-12"}>
                    <div className={"d-flex justify-content-between my-1"}>
                        <h3 style={{fontWeight: fw}}>  Current Price: </h3>
                        <h3 style={{fontWeight: fw}}>  {data.c}</h3>
                    </div>
                    <div className={"d-flex justify-content-between my-1"}>
                        <h3 style={{fontWeight: fw}}> High Price: </h3>
                        <h3 style={{fontWeight: fw}}> {data.h}</h3>
                    </div>
                    <div className={"d-flex justify-content-between my-1"}>
                        <h3 style={{fontWeight: fw}}> Low Price: </h3>
                        <h3 style={{fontWeight: fw}}> {data.l}</h3>
                    </div>
                    <div className={"d-flex justify-content-between my-1"}>
                        <h3 style={{fontWeight: fw}}>  Open Price: </h3>
                        <h3 style={{fontWeight: fw}}> {data.c}</h3>
                    </div>
                    <div className={"d-flex justify-content-between my-1"}>
                        <h3 style={{fontWeight: fw}}>  Prev. Close Price: </h3>
                        <h3 style={{fontWeight: fw}}> {data.pc}</h3>
                    </div>
                    <div className={"d-flex justify-content-between my-1"}>
                        <h3 style={{fontWeight: fw}}>  % Change Today: </h3>
                        <h3 style={{fontWeight: fw}}>  {data.dp.toFixed(2)}%</h3>
                    </div>
                </div>
            </div>
        </Container>
    </>)

}

export default StockStats;