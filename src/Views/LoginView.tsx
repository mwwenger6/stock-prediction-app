import {Container, Form, Button} from "react-bootstrap";

const LoginView = () => {

    return (
        <Container>
            <div className="login">
                <form className="login-form">
                    <label htmlFor="email" style={{textAlign: 'left', padding: '0.25rem 0'}}>Email:</label>
                    <input type="email" id="email" name="email" placeholder="Please enter your email" />
                    <label htmlFor="password" style={{textAlign: 'left', padding: '0.25rem 0'}}>Password:</label>
                    <input type="password" id="password" name="password" />
                    <Button style={{margin: '0.5rem 0'}}variant="primary" type="submit" id="loginButton" name="loginButton" >Login</Button>
                </form>
            </div>
        </Container>
    );
}

export default LoginView;