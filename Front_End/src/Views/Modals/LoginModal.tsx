import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, {Dispatch, SetStateAction, useEffect, useState} from 'react';
import EmailAndPWForm from "../../Components/EmailAndPWForm";
import User from "../../Interfaces/User"
import endpoints from '../../config';

interface LoginModalProps {
    showModal: boolean;
    toggleModal: any;
    showSignUpModal: any;
    setUser: Dispatch<SetStateAction<User | null>>;
}

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
const PWD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%])[A-Za-z\d!@#$%]{8,30}$/;
const SUCCESS: string = "Log In Successful"
const FAILURE: string = "Log In Failed"
const INCORRECT_PASSWORD = "Incorrect Password"
const UNVERIFIED_ACCOUNT = "Verify Your Email Address"

const LoginModal: React.FC<LoginModalProps> = (props: LoginModalProps) => {
    const [email, setEmail] = useState('');
    const [validEmail, setValidEmail] = useState(true)
    const [password, setPassword] = useState('');
    const [validPassword, setValidPassword] = useState(true)
    const [response, setResponse] = useState('');
    const [pendingRequest, setPendingRequest] = useState(false)

    const clearState = () => {
        setEmail('')
        setPassword('')
        setResponse('')
        setValidEmail(false)
        setValidPassword(false)
    }

    useEffect(() => {
        if (!props.showModal) {
            clearState()
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
        setPendingRequest(true)
        const response = await fetch(endpoints.authUser(email, password), {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (response.status == 401) {
            setResponse(INCORRECT_PASSWORD);
            console.error('Error logging in:', response.statusText);
        }
        else if(response.status == 403) {
            setResponse(UNVERIFIED_ACCOUNT);
            console.error('Error logging in:', response.statusText);
        }
        else if (response.status == 200) {
            setResponse(SUCCESS);
            const user: User = await response.json();
            console.log('User logged in: ', user)

            props.setUser(user)
            clearState()
            const timer = setTimeout(() => {
                props.toggleModal()
            }, 500);
        }
        else {
            setResponse(FAILURE);
            console.error('Error logging in:', response.statusText);
        }
        setPendingRequest(false)
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
                    <Button className="btn bg-success border-0 mx-2 " onClick={handleSubmit} disabled={!(validEmail && validPassword) || pendingRequest}>
                        Submit
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
}

export default LoginModal;