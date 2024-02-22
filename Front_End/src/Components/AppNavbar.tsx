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
        <Navbar expand="lg" className="bg-body-tertiary-custom-bg" style={{backgroundColor : '#333333'}}>
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
                        className="me-2"
                        aria-label="Search"
                        value={searchTerm}
                        onChange={handleInputChange}
                        onInput={handleInputSelect} // Added to handle option selection
                        list="tickers-list"
                        style={{ width: '400px' }} // adjust the width as needed
                    />
                    <datalist id="tickers-list">
                        {suggestions.map((suggestion, index) => (
                            <option key={index} value={suggestion.ticker}>{suggestion.name}</option>
                        ))}
                    </datalist>
                </Form>
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ms-auto">
                    <Link to="/" className="nav-link-blue-bg" style={{textDecoration: "none"}} onClick={() => setSearchTerm('')}>Home</Link>
                    <Link to="/News" className="nav-link-blue-bg" style={{textDecoration: "none"}} onClick={() => setSearchTerm('')}>News</Link>
                    {loggedIn ?
                        <>

                            <div className={"d-block"}>
                            <Nav.Link onMouseEnter={() => setIsDropdownOpen(true)} onMouseLeave={() => setIsDropdownOpen(false)} className="nav-link-blue-bg">Account Info</Nav.Link>
                            <Dropdown show={isDropdownOpen} onMouseEnter={() => setIsDropdownOpen(true)} onMouseLeave={() => setIsDropdownOpen(false)}>
                                <Dropdown.Menu>
                                    <Dropdown.Item> <b> Email: </b> {props.user?.email} </Dropdown.Item>
                                    <Dropdown.Item> <b> Account ID: </b> {props.user?.id}</Dropdown.Item>
                                </Dropdown.Menu>
                            </Dropdown>
                            </div>
                            <Nav.Link onClick={() => props.setUser(null)} className="nav-link-blue-bg">Log Out</Nav.Link>
                        </>
                              : <Nav.Link onClick={toggleLogInModal} className="nav-link-blue-bg">Log In</Nav.Link>
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
