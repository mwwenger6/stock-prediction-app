using ServiceStack.DataAnnotations;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("Stocks", Schema = "dbo")]
    public class Stock
    {
#nullable disable
        [Key]
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("Name")]
        public string Name { get; set; }
        [Column("OneDayPercentage")]
        public float OneDayPercentage { get; set; }
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
    }
}
