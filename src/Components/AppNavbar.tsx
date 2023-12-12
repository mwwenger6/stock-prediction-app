import {Container, Nav, Navbar, Form } from "react-bootstrap";
import React, { useState, FormEvent  } from 'react';
import Tickers from '../Data/tickers.json';
import { useNavigate } from 'react-router-dom';
interface Ticker {
    ticker: string;
    name: string;
}

const AppNavbar = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [suggestions, setSuggestions] = useState<Ticker[]>([]);
    const navigate = useNavigate();

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
            navigate(`/Stock/${selectedTicker.ticker}`);
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
                    <Nav.Link href="/" className="nav-link-blue-bg">Home</Nav.Link>
                    <Nav.Link href="/News" className="nav-link-blue-bg">News</Nav.Link>
                    <Nav.Link href="" className="nav-link-blue-bg">Log In</Nav.Link>

                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
}

export default AppNavbar;
