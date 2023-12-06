import {Container} from "react-bootstrap";
import StockGraph from "../Components/StockGraph";
import { useParams } from 'react-router-dom';

const StockGraphView = () => {
    const params = useParams();
    const symbol = params.ticker;


    return (
        <Container>
            <StockGraph/>
        </Container>
    );
}

export default StockGraphView;