
//const API_BASE_URL = 'http://localhost:3002';
// First, import all required modules and components
import FeaturedStock from '../Components/FeaturedStock';
import { useState, useEffect } from "react";
import GetPriceUpdate from "../Services/GetPriceUpdate";
import PersonalGraph from "../Components/PersonalGraph";

// Then, declare any constants or variables
const API_BASE_URL = 'http://18.116.164.159:3002';
// or const API_BASE_URL = 'http://localhost:3002'; for local development


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
    
        /*// Function to fetch stocks data from API
            const fetchStocks = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/stocks`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            // Ensure that the object created in the map function matches the Stock interface
            setFeaturedStockData(data.map((stock: any) => ({
                ...stock,
                // Make sure your API returns the price and up/down information or adjust this mapping
                price: parseFloat(stock.price),
                up: stock.up // Or calculate this based on your own logic
            })));
        } catch (error) {
            console.error('Error fetching stocks:', error);
        }
    };

    fetchStocks();
}, []);*/ // The empty dependency array [] means this effect will run once when the component mounts

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