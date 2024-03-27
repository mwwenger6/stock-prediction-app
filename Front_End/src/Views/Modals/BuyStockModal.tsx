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
}

const BuyStockModal: React.FC<BuyStockModalProps> = (props: BuyStockModalProps) => {
    const [userId, setUserId] = useState(props.user.id);
    const [ticker, setTicker] = useState(props.ticker);
    const  [quantity, setQuantity] = useState(0.0);

    useEffect(() => {
        if (props.userStock !== null) {
            setQuantity(props.userStock.quantity);
        } else {
            setQuantity(0.0);
        }
    }, [props.userStock]);

    const handleSubmit = async (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        event.preventDefault();

        const response = await fetch(endpoints.addUserStock(props.user.id, props.ticker, quantity), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (response.status === 401) {
            console.error('Error logging in:', response.statusText);
        } else if (response.status === 200) {
            const user: User = await response.json();
            console.log('User logged in: ', user);
            setQuantity(0.0);
            const timer = setTimeout(() => {
                props.toggleModal();
            }, 500);
        } else {
            console.error('Error logging in:', response.statusText);
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
                        <BuyStockForm user={props.user} setQuantity={setQuantity} ticker={props.ticker} userStock={props.userStock}/>
                    </div>
                </Modal.Body>
                <Modal.Footer className="justify-content-center d-flex">
                    <Button className="btn bg-success border-0 mx-2 " onClick={handleSubmit}>
                        Submit
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
}

export default BuyStockModal;