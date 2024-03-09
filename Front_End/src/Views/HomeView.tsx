import FeaturedStock from '../Components/FeaturedStock';
import PersonalGraph from "../Components/PersonalGraph";
import User from "../Interfaces/User"
import Stock from "../Interfaces/Stock";
import {FaTimes} from "react-icons/fa";


interface HomeViewProps {
    user: User | null,
    homeviewStocks: Stock[]
}


function HomeView (props : HomeViewProps) {

    const loggedIn = props.user != null;

    return(
        <div className="m-2">
            <div className="row m-3">
                <div className="col-lg-12"> {/* full width for Featured Stocks */}
                    <div className="floatingDiv"> 
                        <h3> {loggedIn ? "Watchlist Stocks" : "Featured Stocks" }</h3>
                        <hr className={"my-1"}/>
                        <div className="featured-stocks-container">
                            {props.homeviewStocks.length === 0 ?
                                <h4 className={"my-3"}> <FaTimes className={"text-danger"}/> Currently No Stocks In Your Watchlist</h4>
                                :
                                <div id="featured-stocks" className="d-flex flex-nowrap overflow-auto featuredStockBg">
                                    <div className="d-flex flex-nowrap">
                                        {props.homeviewStocks.map((stock, index) => (
                                            <FeaturedStock key={index} stock={stock} />
                                        ))}
                                    </div>
                                </div>
                            }
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
                                {props.homeviewStocks.map((stock, index) => (
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