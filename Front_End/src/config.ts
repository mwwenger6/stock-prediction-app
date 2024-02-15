const API_BASE_URL = 'https://stockgenieapi.azurewebsites.net';

const endpoints = {
    authUser: (email : string , password : string) => `${API_BASE_URL}/Home/AuthenticateUser/${email}/${password}`,
    addUser: `${API_BASE_URL}/Home/AddUser`,
    // Add more endpoints as needed
};

export default endpoints;