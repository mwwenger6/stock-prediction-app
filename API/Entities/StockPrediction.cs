using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace Stock_Prediction_API.Entities
{
    [Table("StockPredictions", Schema = "dbo")]
    [PrimaryKey(nameof(Ticker), nameof(PredictedOrder), nameof(CreatedAt))]
    public class StockPrediction
    {
#nullable disable
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("PredictedPrice")]
        public float PredictedPrice { get; set; }
        [Column("PredictedOrder")]
        public int PredictedOrder {  get; set; }
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
    }
}
