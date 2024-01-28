using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services.API_Tools
{
    public class TwelveDataTools(IConfiguration config)
    {
        private readonly HttpClient _httpClient = new();
        private readonly string _apiKey = config.GetValue<string>("APIConfigs:12Data:Key");
        private readonly string _quoteURL = config.GetValue<string>("APIConfigs:12Data:Quote");
        private readonly string _seriesURL = config.GetValue<string>("APIConfigs:12Data:TimeSeries");

        public async Task<string> GetTimeSeriesData(string ticker, string interval, DateTime date)
        {
            string requestUrl = _quoteURL.Replace("{interval}", interval)
                .Replace("{stockSymbol}", ticker).Replace("{apiKey}", _apiKey)
                .Replace("{startDateStr}", date.ToString("MM/dd/yyyy"));
            string jsonData = await GetJSONData(requestUrl);
            return jsonData;
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
