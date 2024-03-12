using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [PrimaryKey(nameof(Ticker), nameof(Time))]
    public class StockPrice
    {
        #nullable disable
            [Column("Ticker")]
            public string Ticker { get; set; }
            [Column("Price")]
            public float Price { get; set; }
            [Column("Time")]
            public DateTime Time { get; set; }
    }

    [Table("StockPrices", Schema = "dbo")]
    public class EODStockPrice : StockPrice
    {
    }

    [Table("StockPrices_5Min", Schema = "dbo")]
    public class FMStockPrice : StockPrice
    {
    }
}
