



namespace Stock_Prediction_API.Services.API_Tools
{
    public class FinnhubAPITools(IConfiguration config)
    {
        private readonly HttpClient _httpClient = new();
        private readonly string _apiKey = config.GetValue<string>("APIConfigs:serverKey");




    }
}
