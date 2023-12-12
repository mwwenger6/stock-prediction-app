
async function GetTimeSeriesData(stockSymbol : string , interval : string) {

    const apiKey = '446a11fe72f149bd881f0753ad465055';

    try{
        const response = await fetch(`https://api.twelvedata.com/time_series?symbol=${stockSymbol}&interval=${interval}&apikey=${apiKey}`)
        const stockData = await response.json();
        return stockData;
    }
    catch (error) {
        console.error('Error:', error);
        return -1;
    }

}
export default GetTimeSeriesData;