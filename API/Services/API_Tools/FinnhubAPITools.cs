using System.Text.Json;

namespace Stock_Prediction_API.Services.API_Tools
{
    public class FinnhubAPITools
    {
        private readonly HttpClient _httpClient = new();
        private readonly string _apiKey;
        private readonly string _quoteURL;

        public FinnhubAPITools(IConfiguration config)
        {
            _apiKey = config.GetValue<string>("APIConfigs:Finnhub:Key");
            _quoteURL = $"https://finnhub.io/api/v1/quote?symbol={{symbol}}&token={_apiKey}";
        }



        public async Task<float> GetRecentPrice(string symbol)
        {
            string requestUrl = _quoteURL.Replace("{symbol}", symbol);
            string jsonData = await GetJSONData(requestUrl);
            var data = JsonSerializer.Deserialize<Dictionary<string, float>>(jsonData);

            if (data.ContainsKey("c"))
            {
                float cValue = data["c"];
                return cValue;
            }
            else
            {
                return -1;
            }
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
