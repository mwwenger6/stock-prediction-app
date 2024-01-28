using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("QuickStocks", Schema = "dbo")]
    public class QuickStock
    {
#nullable disable
        // Assuming Ticker is a primary key and is required
        [Key]
        [Column("Ticker")]
        public string Ticker { get; set; }

    }
}
