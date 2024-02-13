import {FaEye, FaEyeSlash} from "react-icons/fa";
import React, {useEffect, useState} from "react";

interface EmailAndPWFormProps {
    email: string;
    setEmail: React.Dispatch<React.SetStateAction<string>>;
    emailPlaceholder: string;
    password: string;
    setPassword: React.Dispatch<React.SetStateAction<string>>;
    passwordPlaceholder: string;
}

const EmailAndPWForm: React.FC<EmailAndPWFormProps> = (props) => {

    const [showPassword, setShowPassword] = useState(false);

    useEffect(() => {
        setShowPassword(false)
    }, []);

    return (
        <form>
            <label className="form-label h6" htmlFor="email">Email:</label>
            <input type="email"
                   id="email"
                   name="email"
                   value={props.email}
                   onChange={(event) => props.setEmail(event.target.value)}
                   placeholder={props.emailPlaceholder}
                   className="form-control form-control-md w-75 mx-auto"
            />
            <label className="mt-3 form-label h6" htmlFor="password"> Password:</label>
            <div className="input-group w-75 mx-auto">
                <input type={showPassword ? "text" : "password"}
                       id="password"
                       name="password"
                       value={props.password}
                       onChange={(event) => props.setPassword(event.target.value)}
                       placeholder={props.passwordPlaceholder}
                       className="form-control form-control-md"
                />
                <button className="btn btn-outline-secondary" type="button" onClick={() => setShowPassword(!showPassword)}>
                    {showPassword ? <FaEyeSlash /> : <FaEye />}
                </button>
            </div>
        </form>
    )
}

export default EmailAndPWForm;