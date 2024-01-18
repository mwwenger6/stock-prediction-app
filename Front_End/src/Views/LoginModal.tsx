import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, {useState} from 'react';

function LoginModal (props: any) {

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [queryResponse, setQueryResponse] = useState('');

    return (
        <div>
            <Modal show={props.showModal} onHide={props.toggleModal} className="modal-sm">
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
                        <div className={queryResponse === '' ? "text-white" : (queryResponse === 'Account created' ? "text-success" : "text-danger fw-semibold")} >
                            {queryResponse === '' ? 'Placeholder' : (queryResponse === 'Account created' ? queryResponse : '*' + queryResponse)}
                        </div>
                    </div>
                <div className="justify-content-center d-flex">
                    <Button className="bg-success border-0 mx-2">
                         Submit 
                    </Button>
                </div>
                </Modal.Body>
                <Modal.Footer className="justify-content-center d-flex">
                        <a href="/login">Sign Up</a>
                        <a href="/login">Forgot Password?</a>
                </Modal.Footer>
            </Modal>
        </div>
);
}

export default LoginModal;
