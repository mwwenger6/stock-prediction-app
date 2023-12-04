import AppNavbar from "../Components/AppNavbar";
import StockGraph from "../Components/StockGraph";
import FeaturedStock from '../Components/FeaturedStock';
import { useState, useEffect } from "react";

interface Stock {
    name: string;
    ticker: string;
    price: number;
}

function HomeView () {

    const stockData: Stock[] = [
        { name: 'Apple', ticker: 'AAPL', price: 150.25 },
        { name: 'Google', ticker: 'GOOGL', price: 2700.50 },
        { name: 'Amazon', ticker: 'AMZN', price: 3550.75 },
        { name: 'Microsoft', ticker: 'MSFT', price: 340.90 },
        { name: 'Facebook', ticker: 'FB', price: 330.40 },
        { name: 'Tesla', ticker: 'TSLA', price: 950.15 },
        { name: 'Netflix', ticker: 'NFLX', price: 580.60 },
    ];

    const [searchedStock, setSearchedStocks] = useState(null);

    useEffect(() => {
        //probably call api here
        //finhub provides real time stock info that shows the latest price
        //this would probably be a good api to use for specific stock searches
    }, []);

    return(
        <div className="m-2">
            <AppNavbar/>
            <div className="row p-2">
                <div className="my-3 col-lg-4 col-sm-12">
                    <div className="floatingDiv">
                        <h3>Featured Stocks</h3>
                        <hr/>
                        <div className="featured-stocks-container">
                            <div id="featured-stocks" className="d-flex flex-nowrap overflow-auto">
                                {stockData.map((stock, index) => (
                                    <FeaturedStock key={index} stock={stock} />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
                <div className="my-3 col-lg-6 col-sm-12">
                    <div className='floatingDiv'>
                        <h3>Your Stocks</h3>
                        <hr/>
                        <StockGraph />
                    </div>
                </div>
                <div className="my-3 col-lg-2 col-sm-12">
                    <div className="floatingDiv" style={{maxHeight: '500px'}}>
                        <h3>Personal Stocks</h3>
                        <hr/>
                        <div id="personal-stocks" className="overflow-auto" style={{ maxHeight: '400px' }}>
                            <div id="personal-stocks">
                                {stockData.map((stock, index) => (
                                    <div key={index} className="floatingDiv m-auto my-2" style={{ minWidth: '200px', width: '200px' }}>
                                        <p>{stock.ticker}</p>
                                        <p>${stock.price.toFixed(2)}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default HomeView;