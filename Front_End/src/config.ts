const API_BASE_URL = 'https://stockrequests.azurewebsites.net';
const API_PRED_URL = 'https://stockgenieapi.azurewebsites.net';

const endpoints = {

    //User Endpoints
    authUser: (email : string , password : string) => `${API_BASE_URL}/Home/AuthenticateUser/${email}/${password}`,
    addUser: `${API_BASE_URL}/Home/AddUser`,
    getUsers: `${API_BASE_URL}/Home/GetUsers`,
    verifyUser: (code : string) => `${API_BASE_URL}/Home/VerifyUser/${code}`,
    addUserWatchlistStock: (id : number, ticker : string) => `${API_BASE_URL}/Home/AddUserWatchlistStock/${id}/${ticker}`,
    removeUserWatchlistStock: (id : number, ticker : string) => `${API_BASE_URL}/Home/RemoveUserWatchlistStock/${id}/${ticker}`,
    getUserWatchlistStocks: (id : number) => `${API_BASE_URL}/Home/GetUserWatchlistStocks/${id}`,
    addUserStock: (id : number, ticker : string, quantity : number) => `${API_BASE_URL}/User/AddUserStock/${id}/${ticker}/${quantity}`,
    removeUserStock: (id : number, ticker : string) => `${API_BASE_URL}/User/RemoveUserStock/${id}/${ticker}`,

    //Stock Endpoints
    getStocks: `${API_BASE_URL}/Home/GetStocks`,
    getStockData:(ticker : string) => `${API_BASE_URL}/Home/GetStock/${ticker}`,
    getHistStockData: (ticker : string) => `${API_BASE_URL}/Home/GetHistoricalStockData/${ticker}`,
    trainModel: (ticker: string) => `${API_PRED_URL}/Home/TrainModel/${ticker}`,
    getPredictions: (ticker : string) => `${API_BASE_URL}/Home/GetPredictions/${ticker}`,
    predict: (ticker : string, range : number) => `${API_PRED_URL}/Home/AddPrediction/${ticker}`,
    clearPredictions: () => `${API_BASE_URL}/Home/ClearStockPredictions`,
    getOpenMarketDays: (num : number) => `${API_BASE_URL}/Home/GetOpenMarketDays/${num}`,

    //Admin Endpoints
    getErrorLogs: `${API_BASE_URL}/Home/GetErrorLogs`,


    // Add more endpoints as needed
};

export default endpoints;