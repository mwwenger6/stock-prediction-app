
async function GetPriceUpdate(stockSymbol : string) {

    const apiKey = 'cln73dhr01qkjffmt80gcln73dhr01qkjffmt810';

    try{
        const response = await fetch(`https://finnhub.io/api/v1/quote?symbol=${stockSymbol}&token=${apiKey}`)
        const stockData = await response.json();
        return stockData;
    }
    catch (error) {
        console.error('Error:', error);
        return -1;
    }

}
export default GetPriceUpdate;