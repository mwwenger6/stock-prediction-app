using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("UserTypes", Schema = "dbo")]
    public class UserType
    {
#nullable disable
        [Column("UserTypeId")]
        public int Id { get; set; }
        [Column("UserTypeName")]
        public string UserTypeName { get; set; }
        
    }
}
