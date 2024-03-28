using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("VolatileStocks", Schema = "dbo")]
    [PrimaryKey(nameof(Ticker))]
    public class VolatileStock
    {
#nullable disable
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("Name")]
        public string Name { get; set; }
        [Column("Price")]
        public float Price { get; set; }
        [Column("PercentChange")]
        public float PercentChange { get; set; }
        [Column("IsPositive")]
        public bool IsPositive { get; set; }
    }
}
