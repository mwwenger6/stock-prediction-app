import React, { useState } from 'react';
import { FaEye, FaEyeSlash } from 'react-icons/fa'; // Import eye icons
import User from "../Interfaces/User";

interface AccountViewProps {
    user: User | null;
}

const AccountView = (props: AccountViewProps) => {
    const [showPassword, setShowPassword] = useState(false);

    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    return (
        <>
            {props.user != null &&
                <div className={"m-lg-4 m-2 row"}> {/* Adjust layout as needed */}
                    <div className="col-lg-2"> </div>
                    <div className="floatingDiv col-lg-8 col-12">
                        <h1>Account Information</h1>
                        <hr />
                        <div className="row">
                            <div className="col-lg-6 col-12">
                                <div className="d-flex">
                                    <h3>Email:</h3>
                                    <h4 className="fw-light ms-3">{props.user.email}</h4>
                                </div>
                                <div className="d-flex align-items-center">
                                    <h3>Password:</h3>
                                    <h4 className="fw-light ms-3">
                                        {showPassword ? props.user.password : '********'}
                                    </h4>
                                    {showPassword ? <FaEyeSlash className='ms-2 eyeIcon' onClick={togglePasswordVisibility}/> : 
                                                    <FaEye className=' ms-2 eyeIcon' onClick={togglePasswordVisibility}/>} {/* Toggle eye icon */}
                                </div>
                            </div>
                            <div className="col-lg-6 col-12">
                                <div className="d-flex align-items-center">
                                    <h3>Account Type:</h3>
                                    <h4 className="fw-light ms-3">{props.user.typeName}</h4>
                                </div>
                                <div className="d-flex align-items-center">
                                    <h3>Sign Up Date:</h3>
                                    <h4 className="fw-light ms-3">
                                        {new Date(props.user.createdAt).toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'long', day: 'numeric' })}
                                    </h4>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="col-md-2 col-12"> </div>
                </div>
            }
        </>
    );
};

export default AccountView;