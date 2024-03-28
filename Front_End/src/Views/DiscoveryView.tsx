import StockAnalysis from "../Components/StockAnalysis";

const DiscoveryView = () => {

    return (
        <div className={"floatingDiv row m-md-2 m-1"}>
            <h3> Discovery </h3>
            <hr/>
            <div className={"col-12 col-lg-6"}>
                <StockAnalysis/>
            </div>
        </div>
    );
}

export default DiscoveryView;