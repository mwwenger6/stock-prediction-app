
async function GetTimeSeriesData(stockSymbol : string , interval : string) {

    const apiKey = '446a11fe72f149bd881f0753ad465055';
    let startDateStr : string = '';
    let startDate : Date = new Date();

    try{
        if(interval === '5min')
            startDate.setHours(startDate.getHours() - 1);
        else if(interval === '30min')
            startDate.setHours(startDate.getHours() - 24);
        else if(interval === '1day')
            startDate.setMonth(startDate.getMonth() - 1);

        startDateStr = startDate.toLocaleDateString() + ' ' + startDate.toLocaleTimeString();

        let url : string = `https://api.twelvedata.com/time_series?symbol=${stockSymbol}&start_date=${startDateStr}&interval=${interval}&apikey=${apiKey}`;
        const response = await fetch(url)
        const stockData = await response.json();
        return stockData;
    }
    catch (error) {
        console.error('Error:', error);
        return -1;
    }

}
export default GetTimeSeriesData;