import {Container} from "react-bootstrap";
import StockGraph from "../Components/StockGraph";
import { useParams } from 'react-router-dom';
import StockStats from "../Components/StockStats";
import User from "../Interfaces/User";
import Stock from "../Interfaces/Stock";

interface StockGraphViewProps {
    user: User | null,
    featuredStocks: Stock[]
}
const StockGraphView = (props : StockGraphViewProps) => {
    const params = useParams();
    const symbol = params.symbol;
    const isFeatured = props.featuredStocks.some(stock => stock.ticker === symbol);


    return (
        symbol != undefined ?
        <Container>
            <div className="floatingDiv m-4">
                <StockGraph symbol={ symbol } isFeatured={ isFeatured } user = { props.user }/>
                <hr/>
                <StockStats symbol={symbol}/>
            </div>
        </Container>
      : <Container>
                <div className="floatingDiv m-4"> </div>
        </Container>
    );
}

export default StockGraphView;