import StockAnalysis from "../Components/StockAnalysis";
import VolitileStocks from "../Components/VolitileStocks";

const DiscoveryView = () => {

    return (
        <div className={"floatingDiv row m-md-2 m-1"}>
            <h3> Discovery </h3>
            <hr/>
            <div className={"col-12 col-lg-6"}>
                <StockAnalysis/>
            </div>
            <div className={"col-6 col-lg-3"}>
                <VolitileStocks isGainers={true}/>
            </div>
            <div className={"col-6 col-lg-3"}>
                <VolitileStocks isGainers={false}/>
            </div>
        </div>
    );
}

export default DiscoveryView;