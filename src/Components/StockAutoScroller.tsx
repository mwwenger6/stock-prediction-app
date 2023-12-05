import {Container} from "react-bootstrap";
import StockGraph from "../Components/StockGraph";
import { useParams } from 'react-router-dom';

const StockAutoScroller = () => {
    const params = useParams();
    const cikStr = params.cik_str;

    return (
        <Container className = "custom-navbar">
            <h2>Stock Page for CIK: {cikStr}</h2>
            <div>
                This will be Stock Page
            </div>
        </Container>
    );
}

export default StockAutoScroller;