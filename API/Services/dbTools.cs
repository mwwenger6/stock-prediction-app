using Microsoft.EntityFrameworkCore;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    public class dbTools
    {

        private readonly AppDBContext dbContext;

        public dbTools(AppDBContext context)
        {
            dbContext = context;
        }

        public IQueryable<Users> GetUsers()
        {
            return dbContext.Users;
        }
    }
}
