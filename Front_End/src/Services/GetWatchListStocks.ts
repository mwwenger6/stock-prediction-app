import config from "../config";
import Stock from "../Interfaces/Stock";
async function GetWatchListStocks(id : number): Promise<Stock[] | null> {
    try {
        const response = await fetch(config.getUserWatchlistStocks(id));
        const stockList = await response.json();

        if (Array.isArray(stockList)) {
            const stocks: Stock[] = stockList.map((stock: any) => {
                return {
                    name: stock.name,
                    ticker: stock.ticker,
                    price: stock.currentPrice,
                    up: stock.dailyChange > 0
                };
            });
            return stocks;
        }
        else {
            console.error("Error: Invalid data format");
            return null;
        }
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}
export default GetWatchListStocks;