import {Container} from "react-bootstrap";
import StockGraph from "../Components/StockGraph";
import { useParams } from 'react-router-dom';

const StockGraphView = () => {
    const params = useParams();
    const symbol = params.symbol;

    return (
        <Container>
            <div className="floatingDiv m-4">
                <StockGraph symbol={ symbol }/>
            </div>
        </Container>
    );
}

export default StockGraphView;