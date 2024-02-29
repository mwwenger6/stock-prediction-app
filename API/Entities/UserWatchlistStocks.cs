using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("UserWatchlistStocks", Schema = "dbo")]
    public class UserWatchlistStocks
    {
#nullable disable
        [Column("WatchlistId")]
        public int Id { get; set; }
        [Column("UserId")]
        public int UserId { get; set; }
        [Column("Ticker")]
        public string Ticker { get; set; }
    }
}
