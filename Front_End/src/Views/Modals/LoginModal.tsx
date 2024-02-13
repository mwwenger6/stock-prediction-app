import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, { useEffect, useState } from 'react';
import EmailAndPWForm from "../../Components/EmailAndPWForm";

interface User {
    email: string;
    password: string;
    id: number;
    createdAt: string;
}

interface LoginModalProps {
    showModal: boolean;
    toggleModal: any;
    showSignUpModal: any;
}

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
const PWD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%])[A-Za-z\d!@#$%]{8,30}$/;
const SUCCESS: string = "Log In Successful"
const FAILURE: string = "Log In Failed"
const INCORRECT_PASSWORD = "Incorrect Password"

const LoginModal: React.FC<LoginModalProps> = (props: LoginModalProps) => {
    const [email, setEmail] = useState('');
    const [validEmail, setValidEmail] = useState(true)
    const [password, setPassword] = useState('');
    const [validPassword, setValidPassword] = useState(true)
    const [response, setResponse] = useState('');

    useEffect(() => {
        if (!props.showModal) {
            setEmail('')
            setPassword('')
            setResponse('')
            setValidEmail(false)
            setValidPassword(false)
        }
    }, [props.showModal]);

    useEffect(() => {
        setValidEmail(EMAIL_REGEX.test(email))
    }, [email]);

    useEffect(() => {
        setValidPassword(PWD_REGEX.test(password))
    }, [password]);

    const handleSubmit = async (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        event.preventDefault();

        const response = await fetch(`http://localhost:80/Home/AuthenticateUser/${email}/${password}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (response.status == 401) {
            setResponse(INCORRECT_PASSWORD);
            console.error('Error logging in:', response.statusText);
        }
        else if (response.status == 200) {
            setResponse(SUCCESS);
            const user: User = await response.json();
            console.log('User logged in: ', user)
            const timer = setTimeout(() => {
                props.toggleModal()
            }, 500);
        }
        else {
            setResponse(FAILURE);
            console.error('Error logging in:', response.statusText);
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
                        <EmailAndPWForm email={email} setEmail={setEmail} emailPlaceholder={"Enter your username"}
                                        password={password} setPassword={setPassword} passwordPlaceholder={"Enter your password"}/>
                        <div className="my-2">
                            <div className={response === '' ? "text-white" : (response === SUCCESS ? "text-success fw-semibold" : "text-danger fw-semibold")}>
                                {response === '' ? 'Placeholder' : (response === SUCCESS ? response : '*' + response)}
                            </div>
                        </div>
                    </div>
                    <div className="row justify-content-center d-flex">
                        <p className="col-auto mb-1"> Don't have an account?</p>
                        <Button className="col-auto btn-sm" onClick={props.showSignUpModal}>Sign Up</Button>
                    </div>
                    <div className="row justify-content-center d-flex mt-2">
                        <p className="col-auto mb-1"> Forgot Password?</p>
                        <Button className="col-auto btn btn-sm">Reset Password</Button>
                    </div>
                </Modal.Body>
                <Modal.Footer className="justify-content-center d-flex">
                    <Button className="bg-success border-0 mx-2" onClick={handleSubmit} disabled={!(validEmail && validPassword)}>
                        Submit
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
}

export default LoginModal;