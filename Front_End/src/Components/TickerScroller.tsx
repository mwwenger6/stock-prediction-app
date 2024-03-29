import React, {useEffect, useState} from 'react';
import Stock from "../Interfaces/Stock";

interface TickerScrollerProps {
    featuredStocks: Stock[]
}

function TickerScroller (props : TickerScrollerProps) {

    return (
        <div id="stock-ticker" className="ticker-container">
            <div className="scrolling-container">
                {[...Array(100)].map((_, index) => (
                    <div key={index} style={{ display: 'inline-block'}} className={"mx-1"}>
                        {props.featuredStocks.length === 0 ? 
                            <div key={0} className={"stock-item text-black fw-semibold"}>
                                Placeholder: -1
                            </div> :
                        props.featuredStocks.map((stock, idx) => (
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