namespace Stock_Prediction_API.ViewModel
{
#nullable disable
    public class FinnhubStockViewModel
    {
        public string Symbol { get; set; }
        public float CurrentPrice { get; set; }
        public float PercentageChange { get; set; }
        public float OpenPrice { get; set; }
    }
}
