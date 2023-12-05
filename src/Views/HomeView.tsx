import StockGraph from "../Components/StockGraph";
import FeaturedStock from '../Components/FeaturedStock';
import { useState, useEffect } from "react";
import GetPrices from "../Services/PriceUpdate";

interface Stock {
    name: string;
    ticker: string;
    price: number;
}

function HomeView() {

    //Assuming a user's stocks and featured stocks will differ
    const featuredStocks: Stock[] = [
        {name: 'Apple', ticker: 'AAPL', price: 0},
        {name: 'Google', ticker: 'GOOGL', price: 0},
        {name: 'Amazon', ticker: 'AMZN', price: 0},
        {name: 'Microsoft', ticker: 'MSFT', price: 0},
        {name: 'Netflix', ticker: 'NFLX', price: 0},
    ];
    const [personalStocks, setPersonalStocks] = useState<Stock []>([
        {name: 'Thermo Fisher Sci Inc.', ticker: 'TMO', price: 0},
        {name: 'Google', ticker: 'GOOGL', price: 0},
        {name: 'Amazon', ticker: 'AMZN', price: 0},
        {name: 'United Parcel Service Inc', ticker: 'UPS', price: 0},
        {name: 'Walt Disney Co', ticker: 'DIS', price: 0},
    ])

    const [apiStockInfo, setApiStockInfo]: any[] = useState([])

    useEffect(() => {
        const fetchPrices = async () => {
            try {
                const res = await GetPrices(featuredStocks);
                console.log('response retreived', res);
                setApiStockInfo([...apiStockInfo, res]);
                //I am getting the response info here but not setting correctly
            } catch (error) {
                console.error('Error fetching prices:', error);
            }
        };
        fetchPrices();
    }, []);


    return (
        <div className="m-2">
            <div className="row p-2">
                <div className="my-3 col-lg-4 col-sm-12">
                    <div className="floatingDiv">
                        <h3>Featured Stocks</h3>
                        <hr/>
                        <div className="featured-stocks-container">
                            <div id="featured-stocks" className="d-flex flex-nowrap overflow-auto">
                                {featuredStocks.map((stock, index) => (
                                    <FeaturedStock key={index} stock={stock}/>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
                <div className="my-3 col-lg-6 col-sm-12">
                    <div className='floatingDiv'>
                        <h3>Your Stocks</h3>
                        <hr/>
                        <StockGraph/>
                    </div>
                </div>
                <div className="my-3 col-lg-2 col-sm-12">
                    <div className="floatingDiv" style={{maxHeight: '500px'}}>
                        <h3>Personal Stocks</h3>
                        <hr/>
                        <div id="personal-stocks" className="overflow-auto" style={{maxHeight: '400px'}}>
                            <div id="personal-stocks">
                                {personalStocks.map((stock, index) => (
                                    <div key={index} className="floatingDiv m-auto my-2"
                                         style={{minWidth: '200px', width: '200px'}}>
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