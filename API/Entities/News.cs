using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("News", Schema = "dbo")]
    public class News
    {
#nullable disable
        [Column("NewsId")]
        public int Id { get; set; }
        [Column("Title")]
        public string Title { get; set; }
        [Column("Content")]
        public string Content { get; set; }
        [Column("PublishedAt")]
        public DateTime PublishedAt { get; set; }
    }
}
