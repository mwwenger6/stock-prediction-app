import StockGraph from "../Components/StockGraph";
import FeaturedStock from '../Components/FeaturedStock';
import { useState, useEffect } from "react";
import GetPriceUpdate from "../Services/GetPriceUpdate";

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

    return(
        <div className="m-2">
            <div className="row p-2">
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
            <div className="row"> {/* new row for Personal Stocks and Your Stocks */}
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
                    <div className='floatingDiv'>
                        <h3>Your Stocks</h3>
                        <hr/>
                        <StockGraph />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default HomeView;