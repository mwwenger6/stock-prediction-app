const API_BASE_URL = 'https://stockrequests.azurewebsites.net';
const API_PRED_URL = 'https://stockgenieapi.azurewebsites.net';

const endpoints = {
    authUser: (email : string , password : string) => `${API_BASE_URL}/Home/AuthenticateUser/${email}/${password}`,
    addUser: `${API_BASE_URL}/Home/AddUser`,
    getStocks: `${API_BASE_URL}/Home/GetStocks`,
    getStockData:(ticker : string) => `${API_BASE_URL}/Home/GetStock/${ticker}`,
    getErrorLogs: `${API_BASE_URL}/Home/GetErrorLogs`,
    getHistStockData: (ticker : string) => `${API_BASE_URL}/Home/GetHistoricalStockData/${ticker}`,
    trainModel: (ticker: string) => `${API_PRED_URL}/Home/TrainModel/${ticker}`,
    getPredictions: (ticker : string, date : Date) => `${API_BASE_URL}/Home/GetPredictions/${ticker}/${date}`,
    predict: () => `${API_PRED_URL}/Home/AddPredictions`,
    // Add more endpoints as needed
};

export default endpoints;