import FeaturedStock from '../Components/FeaturedStock';
import PersonalGraph from "../Components/PersonalGraph";
import User from "../Interfaces/User"
import Stock from "../Interfaces/Stock";
import {Dispatch, SetStateAction} from "react";


interface HomeViewProps {
    user: User | null,
    setUser: Dispatch<SetStateAction<User | null>>;
    featuredStocks: Stock[]
}


function HomeView (props : HomeViewProps) {

    const loggedIn = props.user != null;

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
                                {props.featuredStocks.map((stock, index) => (
                                    <FeaturedStock key={index} stock={stock} />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {loggedIn ?
                <div className="row m-2"> {/* new row for Personal Stocks and Your Stocks */}
                    <div className="col-lg-2 col-sm-12">
                        <div className="floatingDiv" style={{maxHeight: '500px'}}>
                            <h3>Personal Stocks</h3>
                            <hr/>
                            <div id="personal-stocks" className="overflow-auto" style={{ maxHeight: '400px' }}>
                                {props.featuredStocks.map((stock, index) => (
                                    <div key={index} className="floatingDiv m-auto my-2" style={{ minWidth: '200px', width: '200px' }}>
                                        <p>{stock.ticker}</p>
                                        <p>${stock?.price?.toFixed(2)}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                    <div className="col-lg-10 col-sm-12">
                        <PersonalGraph/>
                    </div>
                </div>
                :
                <div className="row m-4">
                    <div className="floatingDiv">
                        <h3 className="text-center m-3"> Log in/Sign up to access account features</h3>
                    </div>
                </div>
            }
        </div>
    );
}

export default HomeView;