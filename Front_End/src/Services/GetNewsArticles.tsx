
async function GetNewsArticles() {

    var url = 'https://newsapi.org/v2/everything?' +
            'q=Apple&' +
            'from=2024-01-18&' +
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