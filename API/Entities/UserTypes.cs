using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("UserTypes", Schema = "dbo")]
    public class UserType
    {
        [NotMapped] public const string ADMIN = "Admin";
        [NotMapped] public const string CLIENT = "Client";

#nullable disable
        [Column("UserTypeId")]
        public int Id { get; set; }
        [Column("UserTypeName")]
        public string UserTypeName { get; set; }
        
    }
}
