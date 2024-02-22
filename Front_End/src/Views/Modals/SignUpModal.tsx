import React, {useEffect, useState} from "react";
import Modal from "react-bootstrap/Modal";
import EmailAndPWForm from "../../Components/EmailAndPWForm";
import Button from "react-bootstrap/Button";
import endpoints from '../../config';

interface SignUpModalProps {
    showModal: boolean;
    toggleModal: any;
    showLoginModal: any;
}

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
const PWD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%])[A-Za-z\d!@#$%]{8,30}$/;
const SUCCESS: string = "Account Created"
const FAILURE: string = "Failed To Create Account"
const DUPLICATE_EMAIL = "An Account With This Email Already Exists"

const SignUpModal: React.FC<SignUpModalProps> = (props: SignUpModalProps) => {

    const [email, setEmail] = useState('');
    const [validEmail, setValidEmail] = useState(true)
    const [password, setPassword] = useState('');
    const [validPassword, setValidPassword] = useState(true)
    const [response, setResponse] = useState('');
    const [pendingRequest, setPendingRequest] = useState(false)

    const [passwordReqs, setPasswordReqs] = useState({
        hasUppercase: false,
        hasLowercase: false,
        hasNumber: false,
        hasSpecialChar: false,
        isValidLength: false
    });

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

        const hasUppercase = /[A-Z]/.test(password);
        const hasLowercase = /[a-z]/.test(password);
        const hasNumber = /[0-9]/.test(password);
        const hasSpecialChar = /[!@#$%]/.test(password);
        const isValidLength = password.length >= 8 && password.length <= 30;

        setPasswordReqs({
            hasUppercase,
            hasLowercase,
            hasNumber,
            hasSpecialChar,
            isValidLength
        });
    }, [password]);

    const handleSubmit = async (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        event.preventDefault();
        setPendingRequest(true)
        const response = await fetch(endpoints.addUser, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email,
                password
            })
        });

        console.log(response)
        if (response.status == 409) {
            setResponse(DUPLICATE_EMAIL);
            console.error('Error signing up:', response.statusText);
        }
        else if (response.status == 200) {
            setResponse(SUCCESS);
            const timer = setTimeout(() => {
                props.showLoginModal()
            }, 500);
            console.log('Account created with email: ', email)
        }
        else {
            setResponse(FAILURE);
            console.error('Error logging in:', response.statusText);
        }
        setPendingRequest(false)
    }

    return(
        <Modal show={props.showModal} onHide={props.toggleModal} className="modal-md">
            <Modal.Header closeButton>
                <Modal.Title className="w-100 text-center"> Sign Up </Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <div className="w-100 text-center">
                    <EmailAndPWForm email={email} setEmail={setEmail} emailPlaceholder={"Enter a valid username"}
                                    password={password} setPassword={setPassword} passwordPlaceholder={"Enter a valid password"}/>
                    <div className="my-2">
                        <div className={response === '' ? "text-white" : (response === SUCCESS ? "text-success fw-semibold" : "text-danger fw-semibold")}>
                            {response === '' ? 'Placeholder' : (response === SUCCESS ? response : '*' + response)}
                        </div>
                    </div>
                    <h6 className="mt-2"> Password Must... </h6>
                    <div className="d-flex justify-content-center">
                        <ul>
                            <li className={`float-start ${passwordReqs.hasUppercase ? 'text-success' : 'text-danger'}`}>
                                Include at least 1 Uppercase Letter
                            </li>
                            <br/>
                            <li className={`float-start ${passwordReqs.hasLowercase? 'text-success' : 'text-danger'}`}>
                                Include at least 1 Lowercase Letter
                            </li>
                            <br/>
                            <li className={`float-start ${passwordReqs.hasNumber ? 'text-success' : 'text-danger'}`}>
                                Include at least 1 Number
                            </li>
                            <br/>
                            <li className={`float-start ${passwordReqs.hasSpecialChar ? 'text-success' : 'text-danger'}`}>
                                Include at least 1 Special Character (e.g. !@#$%)
                            </li>
                            <br/>
                            <li className={`float-start ${passwordReqs.isValidLength ? 'text-success' : 'text-danger'}`}>
                                Be Between 8 and 30 Characters Long
                            </li>
                        </ul>
                    </div>
                </div>
            </Modal.Body>
            <Modal.Footer className="justify-content-center d-flex">
                <Button className="bg-success border-0 mx-2" onClick={handleSubmit} disabled={!(validEmail && validPassword) || pendingRequest}>
                    Submit
                </Button>
            </Modal.Footer>
        </Modal>
    )
}

export default SignUpModal;