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
    [Table("Stocks", Schema = "dbo")]
    [PrimaryKey(nameof(Ticker))]
    public class Stock
    {
#nullable disable
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("Name")]
        public string Name { get; set; }
#nullable enable
        [Column("OneDayPercentage")]
        public float? OneDayPercentage { get; set; }
#nullable disable
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
    }
}
