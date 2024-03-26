import {FaEye, FaEyeSlash} from "react-icons/fa";
import React, {useEffect, useState} from "react";
import User from "../Interfaces/User";
import UserStock from "../Interfaces/UserStock";

interface BuyStockFormProps {
    ticker: string;
    user: User;
    userStock: UserStock | null;
    setQuantity: React.Dispatch<React.SetStateAction<number>>;
}

const BuyStockForm: React.FC<BuyStockFormProps> = (props) => {

    let quantity = props.userStock ? props.userStock.quantity : 0.0;

    const handleQuantityChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newQuantity = parseFloat(event.target.value); 
        props.setQuantity(isNaN(newQuantity) ? 0.0 : newQuantity); 
    };

    return (
        <form>
            <label className="form-label h6" htmlFor="stock">Stock:</label>
            <input
                   id="stock"
                   name="stock"
                   value={props.ticker}
                   className="form-control form-control-md w-75 mx-auto"
            />
            <label className="mt-3 form-label h6" htmlFor="quantity">Quantity:</label>
            <input 
                    type="number"
                    id="quantity"
                    name="password"
                    value={quantity}
                    onChange={handleQuantityChange}
                    placeholder="0.0"
                    className="form-control form-control-md"
            />
            <input 
                    value={props.user.id}
                    className="hidden"
            />
        </form>
    )
}

export default BuyStockForm;