using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



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
    }
}
