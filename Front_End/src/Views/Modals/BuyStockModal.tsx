import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, {Dispatch, SetStateAction, useEffect, useState} from 'react';
import BuyStockForm from "../../Components/BuyStockForm";
import UserStock from "../../Interfaces/UserStock";
import User from "../../Interfaces/User";
import endpoints from '../../config';

interface BuyStockModalProps {
    showModal: boolean;
    toggleModal: any;
    user: User;
    ticker: string;
    userStock: UserStock | null;
    sellModal: boolean;
}

const BuyStockModal: React.FC<BuyStockModalProps> = (props: BuyStockModalProps) => {
    const [userId, setUserId] = useState(props.user.id);
    const [ticker, setTicker] = useState(props.ticker);
    const  [quantity, setQuantity] = useState(0.0);
    const [paidPrice, setPaidPrice] = useState(0.00);

    useEffect(() => {
        if (props.userStock !== null) {
            setQuantity(props.userStock.quantity);
            setPaidPrice(props.userStock.price);
        } else {
            setQuantity(0.0);
            setPaidPrice(0.00);
        }
    }, [props.userStock]);

    const handleSubmit = async (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        event.preventDefault();
        console.log(props)
        let url = endpoints.addUserStock(props.user.id, props.ticker, quantity, paidPrice);
        if(props.sellModal)
            url = endpoints.subtractUserStock(props.user.id, props.ticker, quantity);
        console.log(url)
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (response.status === 401) {
            console.error('Error buying/selling:', response.statusText);
        } else if (response.status === 200) {
            console.log('Transaction successful');
            setQuantity(0.0);
            const timer = setTimeout(() => {
                props.toggleModal();
            }, 500);
        } else {
            console.error('Error buying/selling:', response.statusText);
        }
    }


    return (
        <div>
            <Modal show={props.showModal} onHide={props.toggleModal} className="modal-md">
                <Modal.Header closeButton>
                    <Modal.Title className="w-100 text-center">  </Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <div className="w-100 text-center">
                        <h3>Change {props.ticker} Shares Owned</h3>
                        <BuyStockForm user={props.user} setQuantity={setQuantity} sellStock={props.sellModal} ticker={props.ticker} userStock={props.userStock} setPrice={setPaidPrice}/>
                    </div>
                </Modal.Body>
                <Modal.Footer className="justify-content-center d-flex">
                    <Button className={`btn ${props.sellModal ? "bg-danger" : "bg-success"} border-0 mx-2 `} onClick={handleSubmit}>
                        Submit
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
}

export default BuyStockModal;