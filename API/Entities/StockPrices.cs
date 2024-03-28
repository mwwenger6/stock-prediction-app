using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("StockPrices", Schema = "dbo")]
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
            [Column("IsClosePrice")]
            public bool IsClosePrice { get; set; }
    }
}
