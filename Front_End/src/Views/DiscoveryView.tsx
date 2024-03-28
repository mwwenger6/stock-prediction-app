import StockAnalysis from "../Components/StockAnalysis";
import VolitileStocks from "../Components/VolitileStocks";
import Stock from "../Interfaces/Stock";
import {FaTimes} from "react-icons/fa";
import FeaturedStock from "../Components/FeaturedStock";

interface DiscoveryViewProps {
    featuredStocks: Stock[],
}
const DiscoveryView = (props : DiscoveryViewProps) => {

    return (
        <div className={"m-md-2 m-1"}>
            <div className="floatingDiv">
                <h3> Stock Genie's Featured Stocks </h3>
                <hr className={"my-1"}/>
                <div className="featured-stocks-container">
                    {props.featuredStocks.length === 0 ?
                        <h4 className={"my-3"}> <FaTimes className={"text-danger"}/> Currently No Featured Stocks </h4>
                        :
                        <div id="featured-stocks" className="d-flex flex-nowrap overflow-auto featuredStockBg">
                            <div className={props.featuredStocks.length === 0 ? "d-flex flex-nowrap mx-auto" : "d-flex flex-nowrap"}>
                                {props.featuredStocks.map((stock, index) => (<FeaturedStock key={index} stock={stock} />))}
                            </div>
                        </div>
                    }
                </div>
            </div>
            <div className={"row m-md-2 m-1"}>
                <div className={"col-6 col-lg-3"}>
                    <div className={'ms-1 ms-md-2 mt-1'}>
                        <VolitileStocks isGainers={true}/>
                    </div>
                </div>
                <div className={"col-6 col-lg-3"}>
                    <div className={'me-1 me-md-2 mt-1'}>
                        <VolitileStocks isGainers={false}/>
                    </div>
                </div>
                <div className={"col-12 col-lg-6 floatingDiv mt-1"}>
                    <StockAnalysis/>
                </div>
            </div>
        </div>
    );
}

export default DiscoveryView;