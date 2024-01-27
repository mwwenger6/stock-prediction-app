using Microsoft.EntityFrameworkCore;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    public class AppDBContext
    {

        public DbSet<Users> Users { get; set; }
        
    }
}
