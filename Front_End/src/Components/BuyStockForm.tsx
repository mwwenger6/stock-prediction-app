import React, {useEffect, useState} from "react";
import User from "../Interfaces/User";
import UserStock from "../Interfaces/UserStock";

interface BuyStockFormProps {
    ticker: string;
    user: User;
    userStock: UserStock | null;
    sellStock: boolean;
    setQuantity: React.Dispatch<React.SetStateAction<number>>;
    setPrice: React.Dispatch<React.SetStateAction<number>>;
}

const BuyStockForm: React.FC<BuyStockFormProps> = (props) => {

    const [quantity, setQuantity] = useState<number>(0.0);
    const [price, setPrice] = useState<number>(0.0);


    const handleQuantityChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newQuantity = parseFloat(event.target.value); 
        setQuantity(isNaN(newQuantity) ? 0.0 : newQuantity);
        props.setQuantity(isNaN(newQuantity) ? 0.0 : newQuantity); 
    };
    const handlePriceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newPrice = parseFloat(event.target.value);
        setPrice(isNaN(newPrice) ? 0.0 : newPrice);
        props.setPrice(isNaN(newPrice) ? 0.0 : newPrice);
    };

    return (
        <form>
            {props.sellStock ? (
                <>
                    <label className="mt-3 form-label h6" htmlFor="quantity">
                        Shares to Sell from total:
                    </label>
                    <input 
                        type="number"
                        id="quantity"
                        name="quantity"
                        value={quantity}
                        onChange={handleQuantityChange}
                        placeholder="0.0"
                        min="0"
                        max={props.userStock ? props.userStock.quantity : 0}
                        className="form-control form-control-md"
                    />
                </>
            ) : (
                <>
                    <label className="mt-3 form-label h6" htmlFor="quantity">
                        Shares to Add to Total:
                    </label>
                    <input 
                        type="number"
                        id="quantity"
                        name="quantity"
                        value={quantity}
                        onChange={handleQuantityChange}
                        placeholder="0.0"
                        min="0"
                        className="form-control form-control-md"
                    />
                </>
            )}
            {!props.sellStock && (
                <>
                    <label className="mt-3 form-label h6" htmlFor="price">
                        Average Price per share:
                    </label>
                    <input
                        type="number"
                        id="price"
                        name="price"
                        value={price}
                        onChange={handlePriceChange}
                        placeholder="0.0"
                        className="form-control form-control-md"
                    />
                </>
            )}
        </form>
    )
}

export default BuyStockForm;