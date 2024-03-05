const API_BASE_URL = 'https://stockrequests.azurewebsites.net';

const endpoints = {
    authUser: (email : string , password : string) => `${API_BASE_URL}/Home/AuthenticateUser/${email}/${password}`,
    addUser: `${API_BASE_URL}/Home/AddUser`,
    getStocks: `${API_BASE_URL}/Home/GetStocks`,
    getStockData:(ticker : string) => `${API_BASE_URL}/Home/GetStock/${ticker}`,
    getErrorLogs: `${API_BASE_URL}/Home/GetErrorLogs`,
    getHistStockData: (ticker : string) => `${API_BASE_URL}/Home/GetHistoricalStockData/${ticker}`,
    trainModel: (ticker: string) => `${API_BASE_URL}/Home/TrainModel/${ticker}`,
    predict: (ticker : string, range : number) => `${API_BASE_URL}/Home/Predict/${ticker}/${range}`,
    // Add more endpoints as needed
};

export default endpoints;