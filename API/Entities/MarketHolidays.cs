using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;
namespace Stock_Prediction_API.Entities
{
    [Table("MarketHolidays", Schema = "dbo")]
    [PrimaryKey(nameof(Day))]
    public class MarketHolidays
    {
        public DateTime Day { get; set; }
    }
}
