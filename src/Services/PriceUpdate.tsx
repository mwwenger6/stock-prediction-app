
interface Stock {
    name: string;
    ticker: string;
    price: number;
}

interface PolygonApiStock{
    T: string; //ticker
    c: number; //close price
    h: number; //highest price
    l: number; //lowest price
    n: number; //number of transactions
    o: number; //open price
    v: number; //trading volume
    vw: number;//volume weighted average
}

const APIKEY = 'y90DLsGi8kbBsrbwpignpEphHpOk0PQd';

function GetStocks(stocks: Stock[] ) {

    async function getStockData(stocks : Stock[]): Promise<any> {
        try {
            const response = await fetch(`https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/2023-11-28?adjusted=true&apiKey=${APIKEY}`);
            const data = response.json();
            let desiredStockData : PolygonApiStock[] = [];
            data.then(res => {
                if(res.results === undefined) return
                stocks.map(stock => {
                    const result = res.results.find((apiStock: PolygonApiStock) => {
                        return apiStock.T == stock.ticker
                    })
                    if(result) desiredStockData.push(result)
                })
            })
            return desiredStockData
        } catch (error) {
            console.error(`Error fetching data`);
            return [];
        }
    }


    if (stocks.length > 0) return getStockData(stocks)

}

export default GetStocks;