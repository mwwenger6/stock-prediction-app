import {Container, Nav, Navbar} from "react-bootstrap";

const AppNavbar = () => {

    return (
        <Navbar expand="lg" className="bg-body-tertiary">
            <Container>
                <Navbar.Brand>Stock Price Prediction App</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <div className="d-flex justify-content-center align-items-center">
                    <input className="form-control mr-sm-2" type="search" placeholder="Search" aria-label="Search" />
                    <button className="btn btn-outline-success my-2 my-sm-0" type="submit">Search</button>
                </div>
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ms-auto">
                        <Nav.Link href="/">Home</Nav.Link>
                        <Nav.Link href="/News">News</Nav.Link>
                        <Nav.Link href="">Log Out</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
}

export default AppNavbar;
