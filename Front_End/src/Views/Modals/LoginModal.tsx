import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, {useEffect, useState} from 'react';
import { FaEye, FaEyeSlash } from 'react-icons/fa';

interface User {
    email: string;
    password: string;
    id: number;
    createdAt: string;
}

function LoginModal(props: any) {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [queryResponse, setQueryResponse] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const SUCCESS: string = "Log In Successful"
    const FAILURE: string = "Log In Failed"
    const INCORRECT_PASSWORD = "Incorrect Password"

    useEffect(() => {
        if (!props.showModal) {
            setEmail('');
            setPassword('');
            setQueryResponse('');
            setShowPassword(false);
        }
    }, [props.showModal]);

    const handleSubmit = async (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        event.preventDefault();
        console.log("Email:", email);
        console.log("Password:", password);
        try {
            const response = await fetch(`http://localhost:80/Home/GetUser/${email}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error('Failed to log in');
            }

            //We can change the way accounts are validated so it is on the server side
            const user: User = await response.json();
            if(user.password == password)
                setQueryResponse(SUCCESS);
            else
                setQueryResponse(INCORRECT_PASSWORD); // Assuming the server sends back a message
        }
        catch (error: any) {
            console.error('Error logging in:', error);
            setQueryResponse(FAILURE);
        }
    }

    return (
        <div>
            <Modal show={props.showModal} onHide={props.toggleModal} className="modal-md">
                <Modal.Header closeButton>
                    <Modal.Title className="w-100 text-center"> Log In </Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <div className="w-100 text-center">
                        <form className="login-form">
                            <label className="form-label" htmlFor="email">Email:</label>
                            <input type="email" id="email" name="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Please enter your email" className="form-control form-control-md w-75" />
                            <label className="mt-3 form-label" htmlFor="password"> Password:</label>
                            <div className="input-group w-75">
                                <input type={showPassword ? "text" : "password"} id="password" name="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Please enter your password" className="form-control form-control-md" />
                                <button className="btn btn-outline-secondary" type="button" onClick={() => setShowPassword(!showPassword)}>
                                    {showPassword ? <FaEyeSlash /> : <FaEye />}
                                </button>
                            </div>
                            <div className="my-2">
                                <div className={queryResponse === '' ? "text-white" : (queryResponse === SUCCESS ? "text-success" : "text-danger fw-semibold")}>
                                    {queryResponse === '' ? 'Placeholder' : (queryResponse === SUCCESS ? queryResponse : '*' + queryResponse)}
                                </div>
                            </div>
                        </form>
                    </div>
                    <div className="row">
                        <div className="row justify-content-center d-flex">
                            <p className="col-auto mb-1"> Don't have an account?</p>
                            <a className="col-auto" href="/login">Sign Up</a>
                        </div>
                        <div className="row justify-content-center d-flex">
                            <p className="col-auto"> Forgot Password?</p>
                            <a className="col-auto" href="/login">Reset Password</a>
                        </div>
                    </div>
                </Modal.Body>
                <Modal.Footer className="justify-content-center d-flex">
                    <Button className="bg-success border-0 mx-2" onClick={handleSubmit}>
                        Submit
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
}

export default LoginModal;