import config from "../config";
async function GetTimeSeriesData(stockSymbol : string , interval : string, marketClosed: boolean, isFeatured: boolean) {

    const apiKey = '446a11fe72f149bd881f0753ad465055';
    let startDateStr : string = '';
    let startDate : Date = new Date();

    function getLastOpenTradingDay() {
        const timeZone = 'America/New_York';
        const currentDate = new Date().toLocaleString('en-US', { timeZone });

        let lastTradingDay = new Date(currentDate);

        if(lastTradingDay.getHours() < 9 || (lastTradingDay.getHours() === 9 && lastTradingDay.getMinutes() < 30))
            lastTradingDay.setDate(lastTradingDay.getDate() - 1);

        while (lastTradingDay.getDay() === 0 || lastTradingDay.getDay() === 6) {
            lastTradingDay.setDate(lastTradingDay.getDate() - 1);
        }

        lastTradingDay.setHours(16, 0, 0, 0);

        return lastTradingDay;
    }

    try{
        //get start date for time series data
        if(marketClosed)
            startDate = getLastOpenTradingDay();

        if(interval === '1min')
            startDate.setHours(startDate.getHours() - 1);

        else if(interval === '5min')
            startDate.setHours(startDate.getHours() - 24*7);

        else
            startDate.setFullYear(startDate.getFullYear() - 1);


        //get data from our API if it is '5min' or '1day' data and it is a featured stock, get data from 12 data otherwise
        let url : string = ''
        let response : Response = new Response();
        let stockData : any = [];
        if(interval === '1min' || !isFeatured){
            startDateStr = startDate.toLocaleDateString() + ' ' + startDate.toLocaleTimeString();
            url = `https://api.twelvedata.com/time_series?symbol=${stockSymbol}&start_date=${startDateStr}&interval=${interval}&apikey=${apiKey}`;
            response = await fetch(url)
            stockData = await response.json();
            return stockData.values;
        }
        else{
            let month : any = startDate.getMonth() + 1; // Months are 0-indexed
            let day : any = startDate.getDate();
            let year : any = startDate.getFullYear();
            month = month < 10 ? '0' + month : month;
            day = day < 10 ? '0' + day : day;
            startDateStr = month + day + year;
            url = config.getStockGraphData(stockSymbol, startDateStr, interval);
            response = await fetch(url)
            stockData = await response.json();
            stockData = stockData.map((item : any) => {
                return {
                    datetime: item.time,
                    high: undefined,
                    open: item.price,
                    close: undefined,
                    volume: undefined
                }
            });
            return stockData;
        }
    }
    catch (error) {
        console.error('Error:', error);
        return -1;
    }

}
export default GetTimeSeriesData;