import {Container} from "react-bootstrap";
import StockGraph from "../Components/StockGraph";
import { useParams } from 'react-router-dom';
import StockStats from "../Components/StockStats";
import User from "../Interfaces/User";
import Stock from "../Interfaces/Stock";
import React, {Dispatch, SetStateAction, useEffect, useState} from "react";
import endpoints from "../config";

interface StockGraphViewProps {
    user: User | null,
    featuredStocks: Stock[],
    watchlistStocks : Stock[]
    reloadWatchlist: () => Promise<void>;
}
const StockGraphView = (props : StockGraphViewProps) => {
    const params = useParams();
    const symbol = params.symbol;
    const isFeatured = props.featuredStocks.some(stock => stock.ticker === symbol);
    const isWatchlist = props.watchlistStocks.some(stock => stock.ticker === symbol);
    const closed = stockMarketClosed();
    const [userStock, setUserStock] = useState(null);

    function stockMarketClosed() {
        const timeZone = 'America/New_York';
        const now = new Date (new Date().toLocaleString('en-US', { timeZone }));

        const dayOfWeek = now.getDay();
        const currentHour = now.getHours();
        const currentMinute = now.getMinutes();

        // Check if it's a weekend (Saturday or Sunday)
        if (dayOfWeek === 0 || dayOfWeek === 6) return true;
        // Check if it's before 9:30 AM or after 4:00 PM ET
        if (currentHour < 9 || (currentHour === 9 && currentMinute < 30) || currentHour >= 16) return true;
        // The stock market is open
        return false;
    }

    useEffect(() => {
        const fetchUserStock = async () => {
            if (props.user != null && symbol != undefined) {
                setUserStock(await fetch(endpoints.getUserStock(props.user.id, symbol)).then(response => response.json()));
                console.log(userStock);
            } else {
                setUserStock(null);
            }
        };

        fetchUserStock();
    }, []);

    return (
        <div className={"m-md-4 m-1"}>
            {symbol != undefined &&
                <div className="row">
                    <div className="col-lg-9 col-12">
                        <div className="floatingDiv mx-2">
                            <StockGraph symbol={ symbol } isFeatured={ isFeatured } user = { props.user } userStock = {userStock} isWatchlist={ isWatchlist } reloadWatchlist = { props.reloadWatchlist } marketClosed={closed} />
                        </div>
                    </div>
                    <div className="col-lg-3 col-12">
                        <div className="mx-2 floatingDiv mt-lg-0 mt-3">
                            <StockStats symbol={symbol}/>
                        </div>
                    </div>
                </div>
            }
        </div>
    );
}

export default StockGraphView;