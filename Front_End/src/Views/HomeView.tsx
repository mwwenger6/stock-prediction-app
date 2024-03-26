import FeaturedStock from '../Components/FeaturedStock';
import PersonalGraph from "../Components/PersonalGraph";
import User from "../Interfaces/User"
import Stock from "../Interfaces/Stock";
import {FaTimes} from "react-icons/fa";
import Spinner from '../Components/Spinner';


interface HomeViewProps {
    user: User | null,
    homeviewStocks: Stock[],
    personalStocks: Stock[]
}


function HomeView (props : HomeViewProps) {

    const loggedIn = props.user != null;

    return(
        <div className="row m-md-2 m-1">
            <div className="col-lg-12"> {/* full width for Featured Stocks */}
                <div className="floatingDiv">
                    <h3> {loggedIn ? "Watchlist Stocks" : "Featured Stocks" }</h3>
                    <hr className={"my-1"}/>
                    <div className="featured-stocks-container">
                        {props.homeviewStocks.length === 0 && loggedIn ?
                            <h4 className={"my-3"}> <FaTimes className={"text-danger"}/> Currently No Stocks In Your Watchlist</h4>
                            :
                            <div id="featured-stocks" className="d-flex flex-nowrap overflow-auto featuredStockBg">
                                <div className={props.homeviewStocks.length === 0 ? "d-flex flex-nowrap mx-auto" : "d-flex flex-nowrap"}>
                                {props.homeviewStocks.length === 0 ? (<div className=""><Spinner size={'large'} height={'120px'} /></div>) : 
                                (props.homeviewStocks.map((stock, index) => (<FeaturedStock key={index} stock={stock} />)))}
                                </div>
                            </div>
                        }
                    </div>
                </div>
            </div>
            {loggedIn ?
                <div className="row"> {/* new row for Personal Stocks and Your Stocks */}
                    <div className="col-lg-2 col-sm-12 mt-2">
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
                    <div className="col-lg-10 col-sm-12 mt-2">
                        <PersonalGraph/>
                    </div>
                </div>
                :
                <div className="col">
                    <div className="floatingDiv my-2">
                        <h3 className="text-center m-3"> Log in/Sign up to access account features</h3>
                    </div>
                </div>
            }
        </div>
    );
}

export default HomeView;