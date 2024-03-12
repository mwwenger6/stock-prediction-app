using System.ComponentModel.DataAnnotations.Schema;

namespace Stock_Prediction_API.Entities
{
    [Table("Users", Schema = "dbo")]
    public class User
    {
#nullable disable
        [Column("UserId")]
        public int Id { get; set; }
        [Column("Email")]
        public string Email { get; set; }
        [Column("Password")]
        public string Password { get; set; }
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; }
        [Column("UserTypeId")]
        public int TypeId { get; set; }
        [Column("UserVerified")]
        public bool IsVerified { get; set; }
#nullable enable
        [Column("VerificationCode")]
        public string? VerificationCode { get; set; }
#nullable disable
        [NotMapped]
        public string TypeName { get; set; }
    }
}
