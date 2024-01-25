
async function GetNewsArticles() {

    var currentDate = new Date();
    currentDate =  new Date(currentDate.getTime() - 86400000);
    const formattedDate = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(currentDate);
    console.log(formattedDate);
    var url = 'https://newsapi.org/v2/everything?' +
            'q=Stock&' +
            `from=${formattedDate}&` +
            'sortBy=popularity&' +
            'apiKey=bfedcfd9c74e469db82924ab09cbba45';

    var req = new Request(url);

    const response = await fetch(req);
    if (response.ok) {
        const data = await response.json();
        return data;

    } else {
        throw new Error(`HTTP error ${response.status}`);
    }

}
export default GetNewsArticles;