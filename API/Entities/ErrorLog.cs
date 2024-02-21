using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("ErrorLogs", Schema = "dbo")]
    public class ErrorLog
    {
#nullable disable
        [Column("ErrorId")]
        public int Id { get; set; }
        [Column("ErrorMessage")]
        public string Message { get; set; }
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
    }
}
