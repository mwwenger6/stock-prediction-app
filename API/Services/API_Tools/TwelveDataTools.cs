using Microsoft.AspNetCore.Http;
using System.Text.Json;
using Stock_Prediction_API.Entities;
using System.Text.Json.Serialization;

namespace Stock_Prediction_API.Services.API_Tools
{
    public class TwelveDataTools
    {
        private readonly HttpClient _httpClient = new();
        private readonly string _apiKey;
        private readonly string _quoteURL;
        private readonly string _seriesURL;
        private readonly string _supportedStocksURL;

        public TwelveDataTools(IConfiguration config)
        {
            _apiKey = config.GetValue<string>("APIConfigs:12Data:Key");
            _quoteURL = config.GetValue<string>("APIConfigs:12Data:Quote");
            _seriesURL = $"https://api.twelvedata.com/time_series?symbol={{symbol}}&interval={{interval}}&apikey={_apiKey}&source=docs&outputsize={{outputSize}}";
            _supportedStocksURL = $"https://api.twelvedata.com/stocks?country=US&type=Common%20Stock&apikey={_apiKey}";
        }
        public class StockData
        {
            [JsonPropertyName("values")]
            public List<StockValue> Values { get; set; }

            [JsonPropertyName("status")]
            public string Status { get; set; }
        }
        public class StockValue
        {
            [JsonPropertyName("datetime")]
            public string Datetime { get; set; }

            [JsonPropertyName("open")]
            public string Open { get; set; }

            [JsonPropertyName("high")]
            public string High { get; set; }

            [JsonPropertyName("low")]
            public string Low { get; set; }

            [JsonPropertyName("close")]
            public string Close { get; set; }

            [JsonPropertyName("volume")]
            public string Volume { get; set; }
        }

        public class SupportedStockData
        {
            [JsonPropertyName("data")]
            public List<SupportedStockValue> Data { get; set; }
        }
        public class SupportedStockValue
        {
            [JsonPropertyName("symbol")]
            public string Symbol { get; set; }

            [JsonPropertyName("name")]
            public string Name { get; set; }

            [JsonPropertyName("currency")]
            public string Currency { get; set; }

            [JsonPropertyName("exchange")]
            public string Exchange { get; set; }

            [JsonPropertyName("mic_code")]
            public string MicCode { get; set; }

            [JsonPropertyName("country")]
            public string Country { get; set; }

            [JsonPropertyName("type")]
            public string Type { get; set; }
        }
        public async Task<List<SupportedStocks>> GetSupportedStockData()
        {
            string jsonData = await GetJSONData(_supportedStocksURL);
            SupportedStockData stockData = JsonSerializer.Deserialize<SupportedStockData>(jsonData);

            List<SupportedStocks> stocks = new List<SupportedStocks>();
            foreach (SupportedStockValue stock in stockData.Data)
            {
                stocks.Add(new()
                {
                    Ticker = stock.Symbol,
                    Name = stock.Name,
                    LastUpdated = DateTime.Now,
                });
            }
            return stocks;
        }

        public async Task<List<StockPrice>> GetTimeSeriesData(string ticker, string interval, string outputSize)
        {
            string requestUrl = _seriesURL.Replace("{interval}", interval)
                .Replace("{symbol}", ticker)
                .Replace("{outputSize}", outputSize);

            string jsonData = await GetJSONData(requestUrl);
            StockData stockData = JsonSerializer.Deserialize<StockData>(jsonData);

            List<StockPrice> prices = new List<StockPrice>();
            foreach(StockValue data in stockData.Values)
            {
                prices.Add(new () {
                    Ticker = ticker,
                    Price = float.Parse(data.Close),
                    Time= DateTime.Parse(data.Datetime),
                });
            }
            return prices;
        }

        public async Task<string> GetPriceForTickers(string tickers, string interval)
        {
            string requestUrl = _quoteURL.Replace("{interval}", interval)
                .Replace("{stockSymbol}", tickers).Replace("{apiKey}", _apiKey);
            string jsonData = await GetJSONData(requestUrl);
            return jsonData;
        }

        public async Task<string> GetJSONData(string requestUrl)
        {
            try
            {
                HttpResponseMessage response = await _httpClient.GetAsync(requestUrl);

                if (response.IsSuccessStatusCode)
                {
                    string jsonContent = await response.Content.ReadAsStringAsync();
                    return jsonContent;
                }
                else
                    throw new HttpRequestException($"Request failed with status code {response.StatusCode}");
            }
            catch (Exception ex)
            {
                throw new Exception($"Error while making the request: {ex.Message}");
            }
        }
    }
}
