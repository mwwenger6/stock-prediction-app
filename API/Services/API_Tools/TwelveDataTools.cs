using Microsoft.AspNetCore.Http;
using System.Text.Json;
using Stock_Prediction_API.Entities;
using System.Text.Json.Serialization;

namespace Stock_Prediction_API.Services.API_Tools
{
    public class TwelveDataTools(IConfiguration config)
    {
        private readonly HttpClient _httpClient = new();
        private readonly string _apiKey = config.GetValue<string>("APIConfigs:12Data:Key");
        private readonly string _quoteURL = config.GetValue<string>("APIConfigs:12Data:Quote");
        private readonly string _seriesURL = $"https://api.twelvedata.com/time_series?symbol={{symbol}}&interval={{interval}}&apikey=446a11fe72f149bd881f0753ad465055&source=docs&outputsize={{outputSize}}";
        public class StockValue
        {
            [JsonPropertyName("datetime")]
            public DateTime Datetime { get; set; }

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
        public class StockData
        {
            [JsonPropertyName("values")]
            public List<StockValue> Values { get; set; }

            [JsonPropertyName("status")]
            public string Status { get; set; }
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
                prices.Add(new StockPrice() {
                    Ticker = ticker,
                    Price = float.Parse(data.Close),
                    Time= data.Datetime,
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
