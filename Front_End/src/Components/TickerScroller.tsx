import React, {useEffect, useState} from 'react';
import Stock from "../Interfaces/Stock";

interface TickerScrollerProps {
    featuredStocks: Stock[]
}

function TickerScroller (props : TickerScrollerProps) {

    const [tickers, setTickers] = useState(props.featuredStocks)

    return (
        <div id="stock-ticker" className="ticker-container">
            <div className="scrolling-container">
                {[...Array(4)].map((_, index) => (
                    <div key={index} style={{ display: 'inline-block'}}>
                        {tickers.map((stock, idx) => (
                            <div key={idx} className={stock.up ? "stock-item text-green fw-semibold" : "stock-item text-red fw-semibold"}>
                                {stock.ticker}: {stock.price}
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default TickerScroller;