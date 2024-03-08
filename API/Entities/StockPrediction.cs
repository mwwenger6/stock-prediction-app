using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace Stock_Prediction_API.Entities
{
    [Table("StockPredictions", Schema = "dbo")]
    [PrimaryKey(nameof(Ticker), nameof(PredictedPrice), nameof(PredictionOrder))]
    public class StockPrediction
    {
#nullable disable
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("PredictedPrice")]
        public float PredictedPrice { get; set; }
        [Column("PredictionOrder")]
        public int PredictionOrder {  get; set; }
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
    }
}
