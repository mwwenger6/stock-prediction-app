namespace Stock_Prediction_API.ViewModel
{
#nullable disable
    public class TwelveDataViewModel
    {
        public string Symbol { get; set; }
        public string Name { get; set; }
        public string Exchange { get; set; }
        public string MicCode { get; set; }
        public string Currency { get; set; }
        public DateTime DateTime { get; set; }
        public long Timestamp { get; set; }
        public string Open { get; set; }
        public string High { get; set; }
        public string Low { get; set; }
        public string Close { get; set; }
        public long Volume { get; set; }
        public string PreviousClose { get; set; }
        public string Change { get; set; }
        public string PercentChange { get; set; }
        public long AverageVolume { get; set; }
        public string Rolling1dChange { get; set; }
        public string Rolling7dChange { get; set; }
        public string RollingPeriodChange { get; set; }
        public bool IsMarketOpen { get; set; }
        public FiftyTwoWeekData FiftyTwoWeek { get; set; }
        public string ExtendedChange { get; set; }
        public string ExtendedPercentChange { get; set; }
        public string ExtendedPrice { get; set; }
        public long ExtendedTimestamp { get; set; }
    }

    public class FiftyTwoWeekData
    {
        public string Low { get; set; }
        public string High { get; set; }
        public string LowChange { get; set; }
        public string HighChange { get; set; }
        public string LowChangePercent { get; set; }
        public string HighChangePercent { get; set; }
        public string Range { get; set; }
    }
}
