import FeaturedStock from '../Components/FeaturedStock';
import StockAutoscroller from "../Components/StockAutoScroller";
import { useState, useEffect } from "react";
import GetPriceUpdate from "../Services/GetPriceUpdate";
import PersonalGraph from "../Components/PersonalGraph";

interface Stock {
    name: string;
    ticker: string;
    price: number;
    up?: boolean;
}

const initialFeaturedStockData: Stock[] = [
    { name: 'Apple', ticker: 'AAPL', price: -1, up: undefined},
    { name: 'Google', ticker: 'GOOGL', price: -1, up: undefined},
    { name: 'Amazon', ticker: 'AMZN', price: -1, up: undefined },
    { name: 'Microsoft', ticker: 'MSFT', price: -1, up: undefined },
    { name: 'Meta Platforms', ticker: 'META', price: -1, up: undefined },
    { name: 'Tesla', ticker: 'TSLA', price: -1, up: undefined },
    { name: 'Netflix', ticker: 'NFLX', price: -1, up: undefined },
];

function HomeView () {

    const getPrice = GetPriceUpdate;
    const stockData: Stock[] = [
        { name: 'Apple', ticker: 'AAPL', price: 150.25 },
        { name: 'Google', ticker: 'GOOGL', price: 2700.50 },
        { name: 'Amazon', ticker: 'AMZN', price: 3550.75 },
        { name: 'Microsoft', ticker: 'MSFT', price: 340.90 },
        { name: 'Facebook', ticker: 'FB', price: 330.40 },
        { name: 'Tesla', ticker: 'TSLA', price: 950.15 },
        { name: 'Netflix', ticker: 'NFLX', price: 580.60 },
    ];

    const [featuredStockData, setFeaturedStockData] = useState(initialFeaturedStockData)
    const [searchedStock, setSearchedStocks] = useState(null);

    useEffect(() => {

        //Fetch price data on load
        const fetchData = async () => {
            try {
                const updatedStockData  = await Promise.all(featuredStockData.map(async (stock) => {
                    const stockData = await getPrice(stock.ticker);
                    return { ...stock, price: stockData.c, up: stockData.dp > 0 }
                }));

                setFeaturedStockData(updatedStockData)
            }
            catch (error) {
                console.error('Error fetching prices:', error);
            }
        };

        fetchData();
    }, []);

    // useEffect(() => {
    //     const Alpaca = require("@alpacahq/alpaca-trade-api");
    //     const alpaca = new Alpaca({
    //         keyId: 'PK4K7YJE7Z04K0FVFZOL',
    //         secretKey: 'w6ByJtvSkqLOnNPcI42YaKbEZHXfEHsgbj1nOSl9',
    //         paper: true,
    //       })
    //       console.log('hello')
    //       alpaca.getAccount().then((account : any) => {
    //         console.log('Current Account:', account)
    //       })    
    // }, []);

    return(
        <div className="m-2">
            <div className="row m-3">
                <div className="col-lg-12"> {/* full width for Featured Stocks */}
                    <div className="floatingDiv"> 
                        <h3>Featured Stocks</h3>
                        <hr/>
                        <div className="featured-stocks-container">
                            <div id="featured-stocks" className="d-flex flex-nowrap overflow-auto">
                                {featuredStockData.map((stock, index) => (
                                    <FeaturedStock key={index} stock={stock} />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="row m-2"> {/* new row for Personal Stocks and Your Stocks */}
                <div className="col-lg-2 col-sm-12"> {/* adjust the size if u want */}
                    <div className="floatingDiv" style={{maxHeight: '500px'}}>
                        <h3>Personal Stocks</h3>
                        <hr/>
                        <div id="personal-stocks" className="overflow-auto" style={{ maxHeight: '400px' }}>
                            {featuredStockData.map((stock, index) => (
                                <div key={index} className="floatingDiv m-auto my-2" style={{ minWidth: '200px', width: '200px' }}>
                                    <p>{stock.ticker}</p>
                                    <p>${stock.price.toFixed(2)}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
                <div className="col-lg-10 col-sm-12"> {/* adjust the size if u want */}
                    <PersonalGraph/>
                </div>
            </div>
        </div>
    );
}

export default HomeView;
