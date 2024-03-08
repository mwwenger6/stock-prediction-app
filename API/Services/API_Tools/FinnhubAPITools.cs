using System.Text.Json;
using Stock_Prediction_API.Entities;
using static Stock_Prediction_API.Services.API_Tools.FinnhubAPITools.FinnhubMarketHolidays;

namespace Stock_Prediction_API.Services.API_Tools
{
    public class FinnhubAPITools
    {
        private readonly HttpClient _httpClient = new();
        private readonly string _apiKey;
        private readonly string _quoteURL;
        private readonly string _statusURL;
        private readonly string _holidaysURL;

        public FinnhubAPITools(IConfiguration config)
        {
            _apiKey = config.GetValue<string>("APIConfigs:Finnhub:Key");
            _quoteURL = $"https://finnhub.io/api/v1/quote?symbol={{symbol}}&token={_apiKey}";
            _statusURL = $"https://finnhub.io/api/v1/stock/market-status?exchange=US&token={_apiKey}";
            _holidaysURL = $"https://finnhub.io/api/v1/stock/market-holiday?exchange=US&token={_apiKey}";
        }

        public class FinnhubMarketStatus
        {
            public bool IsOpen { get; set; }
        }
        public class FinnhubMarketHolidays
        {
            public List<Holiday> data { get; set; }
            public class Holiday
            {
                public DateTime atDate { get; set; }
                public string tradingHour { get; set; }
            }
        }

        public async Task<Stock> GetRecentPrice(string symbol)
        {
            string requestUrl = _quoteURL.Replace("{symbol}", symbol);
            string jsonData = await GetJSONData(requestUrl);
            var data = JsonSerializer.Deserialize<Dictionary<string, float>>(jsonData);

            Stock stock = new Stock();
            if (data.ContainsKey("c") && data.ContainsKey("dp"))
            {
                 stock.CurrentPrice = data["c"];

                 stock.DailyChange = data["dp"];
                return stock;
            }
            else
            {
                return null;
            }
        }

        public async Task<bool?> GetIsMarketOpen()
        {
            string jsonData = await GetJSONData(_statusURL);
            var status = JsonSerializer.Deserialize<FinnhubMarketStatus>(jsonData);
            return status?.IsOpen;
        }

        public async Task<List<MarketHolidays>> GetMarketHolidays()
        {
            DateTime currentDate = DateTime.Now;
            string jsonData = await GetJSONData(_holidaysURL);
            FinnhubMarketHolidays holidays = JsonSerializer.Deserialize<FinnhubMarketHolidays>(jsonData);
            List<MarketHolidays> list = new List<MarketHolidays>();
            foreach(Holiday h in holidays.data)
            {
                if(h.tradingHour.Count() == 0 && h.atDate >= currentDate)
                {
                    list.Add(new() { Day = h.atDate });
                }
            }
            return list;
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
