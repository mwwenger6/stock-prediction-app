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
    [Table("SupportedStocks", Schema = "dbo")]
    [PrimaryKey(nameof(Ticker))]
    public class SupportedStocks
    {
        [Column("Ticker")]
        public string Ticker { get; set; }
        [Column("Name")]
        public string Name { get; set; }
        [Column("LastUpdated")]
        public DateTime LastUpdated { get; set; }
    }
}