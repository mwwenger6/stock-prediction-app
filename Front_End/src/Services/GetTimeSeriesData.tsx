async function GetTimeSeriesData(stockSymbol : string , interval : string, marketClosed: boolean) {

    const apiKey = '446a11fe72f149bd881f0753ad465055';
    let startDateStr : string = '';
    let startDate : Date = new Date();

    function getLastOpenTradingDay() {
        const timeZone = 'America/New_York';
        const currentDate = new Date().toLocaleString('en-US', { timeZone });

        let lastTradingDay = new Date(currentDate);

        while (lastTradingDay.getDay() === 0 || lastTradingDay.getDay() === 6) {
            lastTradingDay.setDate(lastTradingDay.getDate() - 1);
        }

        lastTradingDay.setHours(16, 0, 0, 0);

        return lastTradingDay;
    }

    try{
        let priceInterval: string = '';

        if(marketClosed)
            startDate = getLastOpenTradingDay();

        if(interval === '1 Hour'){
            priceInterval = '1min';
            startDate.setHours(startDate.getHours() - 1);
        }
        else if(interval === '1 Day'){
            priceInterval = '5min';
            startDate.setHours(startDate.getHours() - 24);
        }
        else if(interval === '1 Week'){
            priceInterval = '5min';
            startDate.setHours(startDate.getHours() - 24*7);
        }
        else if(interval === '1 Month'){
            priceInterval = '1day';
            startDate.setMonth(startDate.getMonth() - 1);
        }
        else if(interval === '1 Year'){
            priceInterval = '1day';
            startDate.setFullYear(startDate.getFullYear() - 1);
        }

        startDateStr = startDate.toLocaleDateString() + ' ' + startDate.toLocaleTimeString();

        let url : string = `https://api.twelvedata.com/time_series?symbol=${stockSymbol}&start_date=${startDateStr}&interval=${priceInterval}&apikey=${apiKey}`;
        const response = await fetch(url)
        const stockData = await response.json();
        console.log(stockData);
        return stockData;
    }
    catch (error) {
        console.error('Error:', error);
        return -1;
    }

}
export default GetTimeSeriesData;