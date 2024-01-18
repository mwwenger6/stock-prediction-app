import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, {useState} from 'react';

function LoginModal (props: any) {

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showError, setShowError] = useState('');

    return (
        <div>
            <Modal show={props.showModal} onHide={props.toggleModal} className="modal-lg">
                <Modal.Header closeButton>
                    <Modal.Title className="w-100 text-center"> Log In </Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <div className="w-100 text-center">
                        <form className="login-form">
                            <label htmlFor="email">Email:</label>
                            <input type="email" id="email" name="email" placeholder="Please enter your email" />
                            <label htmlFor="password"> Password:</label>
                            <input type="password" id="password" name="password" placeholder="Please enter your password"/>
                        </form>
                        <div className={showError? "text-danger fw-semibold" : "text-white"} > 
                            <p> Invalid Username/Password </p>
                        </div>
                    </div>
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="danger" onClick={props.toggleModal}>
                        Close
                    </Button>
                    <Button className="bg-success border-0 mx-2">
                         Submit 
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
);
}

export default LoginModal;
