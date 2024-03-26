const API_BASE_URL = 'https://stockrequests.azurewebsites.net';
const API_PRED_URL = 'https://stockgenieapi.azurewebsites.net';

const endpoints = {

    //User Endpoints
    authUser: (email : string , password : string) => `${API_BASE_URL}/User/AuthenticateUser/${email}/${password}`,
    addUser: `${API_BASE_URL}/User/AddUser`,
    getUsers: `${API_BASE_URL}/User/GetUsers`,
    verifyUser: (code : string) => `${API_BASE_URL}/User/VerifyUser/${code}`,
    addUserWatchlistStock: (id : number, ticker : string) => `${API_BASE_URL}/User/AddUserWatchlistStock/${id}/${ticker}`,
    removeUserWatchlistStock: (id : number, ticker : string) => `${API_BASE_URL}/User/RemoveUserWatchlistStock/${id}/${ticker}`,
    getUserWatchlistStocks: (id : number) => `${API_BASE_URL}/User/GetUserWatchlistStocks/${id}`,
    addUserPersonalStock: (userId : number, ticker : string, quantity : number) => `${API_BASE_URL}/User/AddUserStock/${userId}/${ticker}/${quantity}`,
    getUserStocks: (userId : number) => `${API_BASE_URL}/User/GetUserStocks/${userId}`,
    getUserStockData: (userId : number) => `${API_BASE_URL}/User/GetUserStockData/${userId}`,

    //Stock Endpoints
    getStocks: `${API_BASE_URL}/Stock/GetStocks`,
    getStockData:(ticker : string) => `${API_BASE_URL}/Stock/GetStock/${ticker}`,
    getStockGraphData: (ticker : string, startDate : string, interval : string) => `${API_BASE_URL}/Stock/GetStockGraphData/${ticker}/${startDate}/${interval}`,
    getOpenMarketDays: (num : number) => `${API_BASE_URL}/Stock/GetOpenMarketDays/${num}`,
    getSupportedStocks: () => `${API_BASE_URL}/Stock/GetSupportedStocks`,

    //Prediction Endpoints
    trainModel: (ticker: string) => `${API_PRED_URL}/Prediction/TrainModel/${ticker}`,
    getPredictions: (ticker : string) => `${API_PRED_URL}/Prediction/GetPredictions/${ticker}`,
    predict: (ticker : string, range : number) => `${API_PRED_URL}/Prediction/AddPredictions/${ticker}`,
    clearPredictions: () => `${API_PRED_URL}/Prediction/ClearPredictions`,

    //Admin Endpoints
    getErrorLogs: `${API_BASE_URL}/Admin/GetErrorLogs`,

    // Add more endpoints as needed
};

export default endpoints;