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
    [Table("UserStocks", Schema = "dbo")]
    [PrimaryKey(nameof(UserId), nameof(Ticker))]
    public class UserStock
    {
#nullable disable
        [Column("UserId")]
        public int UserId { get; set; }
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("Quantity")]
        public float Quantity { get; set; }
        [Column("Price")]
        public float Price { get; set; }
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
    }
}
