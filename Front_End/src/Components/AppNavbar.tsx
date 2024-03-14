import {Container, Nav, Navbar, Form, Dropdown } from "react-bootstrap";
import React, {Dispatch, SetStateAction, useState} from 'react';
import Tickers from '../Data/tickers.json';
import {Link, useNavigate} from 'react-router-dom';
import LoginModal from "../Views/Modals/LoginModal";
import SignUpModal from "../Views/Modals/SignUpModal";
import Ticker from "../Interfaces/Ticker";
import User from "../Interfaces/User";
import HomeView from "../Views/HomeView";
import App from "../App";


interface AppNavbarProps {
    user: User | null
    setUser: Dispatch<SetStateAction<User | null>>;
}

const AppNavbar = (props: AppNavbarProps) => {
    const loggedIn = props.user != null;
    const linkClasses = "nav-link-custom border-lg-end";
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [showLoginModal , setShowLoginModal] = useState(false);
    const [showSignUpModal, setShowSignUpModal] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [suggestions, setSuggestions] = useState<Ticker[]>([]);
    const navigate = useNavigate();

    const toggleLogInModal = () => {
        setShowLoginModal(!showLoginModal);
    };

    const toggleSignUpModal = () => {
        setShowSignUpModal(!showSignUpModal);
      };

    const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        setSearchTerm(value);
        if (value.length > 0) {
            const filteredSuggestions = Tickers.filter(
                ticker => ticker.ticker.toLowerCase().includes(value.toLowerCase()) ||
                          ticker.name.toLowerCase().includes(value.toLowerCase())
            ).slice(0, 5); // Limiting to 5 suggestions
            setSuggestions(filteredSuggestions);
        } else
            setSuggestions([]);
    };
    const handleInputSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        setSearchTerm(value); 
        const selectedTicker = suggestions.find(suggestion => suggestion.ticker.toLowerCase() === value.toLowerCase());
        if (selectedTicker) {
            navigate(`/Stock/${selectedTicker.ticker}`, {replace: true})
        }
    };
    return (
        <Navbar expand="lg" className="bg-body-tertiary-custom-bg" style={{backgroundColor : '#333333', marginTop : '0 !important'}}>
            <Container>
                <Navbar.Brand>
                    <span className="brand-name">stock</span>
                    <span className="brand-name-secondary">Genie</span>
                </Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Form className="d-flex" onSubmit={(event) => event.preventDefault()}>
                    <Form.Control
                        type="search"
                        placeholder="Search"
                        className="me-2 mb-lg-0 mb-1"
                        aria-label="Search"
                        value={searchTerm}
                        onChange={handleInputChange}
                        onInput={handleInputSelect} // Added to handle option selection
                        list="tickers-list"
                        style={{ width: '300px'}} // adjust the width as needed
                    />
                    <datalist id="tickers-list">
                        {suggestions.map((suggestion, index) => (
                            <option key={index} value={suggestion.ticker}>{suggestion.name}</option>
                        ))}
                    </datalist>
                </Form>
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ms-auto">
                        <Link to="/" className={linkClasses + " border-lg-start" } onClick={() => setSearchTerm('')}>Home</Link>
                        <Link to="/News" className={linkClasses} onClick={() => setSearchTerm('')}>News</Link>
                        {loggedIn ?
                        <>
                            <div className={"d-block"}>
                                <Link to="/" onClick={(e) =>{
                                    e.preventDefault(); //prevents page navigation
                                    setIsDropdownOpen(!isDropdownOpen)
                                }} className={linkClasses} >Settings</Link>
                                <Dropdown show={isDropdownOpen}>
                                    <Dropdown.Menu>
                                        <Dropdown.Item> <Link to='/Settings/Account' className="text-black" style={{textDecoration: 'none'}} onClick={() => setIsDropdownOpen(false)}>
                                            My Account
                                        </Link></Dropdown.Item>
                                        {props.user && props.user.typeId === 1 && (
                                            <Dropdown.Item> <Link to="/Settings/Admin" className="text-black" style={{textDecoration: 'none'}}  onClick={() => setIsDropdownOpen(false)}>
                                                Admin Control
                                            </Link></Dropdown.Item>
                                        )}
                                    </Dropdown.Menu>
                                </Dropdown>
                            </div>
                            <Link to="/" onClick={() => props.setUser(null)} className={linkClasses} >Log Out</Link>
                        </>
                        : <Link to="/" onClick={(e) =>{
                            e.preventDefault(); //prevents page navigation
                            toggleLogInModal();
                        }} className={linkClasses} >Log In</Link>
                        }
                    </Nav>
                </Navbar.Collapse>
                <LoginModal showModal={showLoginModal} toggleModal={toggleLogInModal} setUser={props.setUser} showSignUpModal={() => {
                    toggleSignUpModal();
                    toggleLogInModal();
                }}/>
                <SignUpModal showModal={showSignUpModal} toggleModal={toggleSignUpModal} showLoginModal={() => {
                    toggleSignUpModal();
                    toggleLogInModal();
                }}/>
            </Container>
        </Navbar>
    );
}   

export default AppNavbar;
